#include "mb/system/MultibodySystem.h"
#include "mb/solvers/DirectSolver.h"
#include "mb/integrators/RungeKutta.h"
#include "mb/integrators/BDF.h"
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <numeric>

namespace mb {

MultibodySystem::MultibodySystem(const std::string& name)
    : name_(name) {}

// ---- Building ----

void MultibodySystem::addBody(std::shared_ptr<Body> body) {
    bodies_.push_back(body);
    initialized_ = false;
}

void MultibodySystem::addConstraint(std::shared_ptr<Constraint> constraint) {
    constraints_.push_back(constraint);
    initialized_ = false;
}

void MultibodySystem::addForce(std::shared_ptr<Force> force) {
    forces_.push_back(force);
}

// ---- Dynamic-only index tracking (mirrors TS dynamicBodyList/dynamicVOffsets) ----

void MultibodySystem::rebuildDynamicIndices() {
    dynBodyIndices_.clear();
    dynVOffsets_.clear();
    dynBodyIdToIdx_.clear();
    totalDynNv_ = 0;

    for (int b = 0; b < (int)bodies_.size(); b++) {
        if (bodies_[b]->isDynamic()) {
            dynBodyIdToIdx_[bodies_[b]->id] = (int)dynBodyIndices_.size();
            dynBodyIndices_.push_back(b);
            dynVOffsets_.push_back(totalDynNv_);
            totalDynNv_ += state_.nvPerBody[b];
        }
    }
}

// ---- State management ----

void MultibodySystem::initialize() {
    if (bodies_.empty()) return;

    auto ptrs = bodyPtrs();
    int nc = numConstraintEquations();
    state_ = StateVector::fromBodies(ptrs, nc);

    rebuildDynamicIndices();

    // Default solver/integrator if not set
    if (!solver_)
        solver_ = std::make_shared<DirectSolver>();
    if (!integrator_)
        integrator_ = std::make_shared<RungeKutta4>();

    initialized_ = true;

    bool isHHT = (dynamic_cast<HHTAlpha*>(integrator_.get()) != nullptr);
    // HHT: mild Baumgarte=5 evaluated at n+1 state (via solveKKTAtHHTState).
    //   γ(q_{n+1},v_{n+1}) provides gentle position/velocity feedback each step.
    // Explicit: Baumgarte=20 embedded in f() — robust drift control.
    const double baumgarte = isHHT ? 50.0 : 20.0;
    for (auto& c : constraints_)
        c->setBaumgarteParameters(baumgarte, baumgarte);
}

void MultibodySystem::setState(const StateVector& st) {
    state_ = st;
    syncStateToBodie();
}

// ---- Assembly (DYNAMIC BODIES ONLY — matches TS) ----

MatrixN MultibodySystem::assembleMassMatrix() const {
    MatrixN M(totalDynNv_, totalDynNv_);
    int nDyn = (int)dynBodyIndices_.size();

    #pragma omp parallel for schedule(static)
    for (int di = 0; di < nDyn; di++) {
        int b = dynBodyIndices_[di];
        int off = dynVOffsets_[di];
        int n = state_.nvPerBody[b];

        auto block = bodies_[b]->computeMassBlock();
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                M.set(off + i, off + j, block[i * n + j]);
    }
    return M;
}

MatrixN MultibodySystem::assembleJacobian() const {
    int nc = numConstraintEquations();
    if (nc == 0) return MatrixN(0, totalDynNv_);

    MatrixN Cq(nc, totalDynNv_);
    int row = 0;
    for (auto& c : constraints_) {
        auto jac = c->computeJacobian();
        auto bodyIds = c->getBodyIds();
        int neq = c->numEquations();

        // Body 1 — only include if dynamic
        if (bodyIds.size() >= 1) {
            auto it = dynBodyIdToIdx_.find(bodyIds[0]);
            if (it != dynBodyIdToIdx_.end()) {
                int di = it->second;
                int off = dynVOffsets_[di];
                int n = state_.nvPerBody[dynBodyIndices_[di]];
                for (int i = 0; i < neq; i++)
                    for (int j = 0; j < std::min(n, jac.J1.cols); j++)
                        Cq.set(row + i, off + j, jac.J1.get(i, j));
            }
            // else: body is static/ground → skip columns (they stay zero)
        }

        // Body 2 — only include if dynamic
        if (bodyIds.size() >= 2) {
            auto it = dynBodyIdToIdx_.find(bodyIds[1]);
            if (it != dynBodyIdToIdx_.end()) {
                int di = it->second;
                int off = dynVOffsets_[di];
                int n = state_.nvPerBody[dynBodyIndices_[di]];
                for (int i = 0; i < neq; i++)
                    for (int j = 0; j < std::min(n, jac.J2.cols); j++)
                        Cq.set(row + i, off + j, jac.J2.get(i, j));
            }
        }

        row += neq;
    }
    return Cq;
}

std::vector<double> MultibodySystem::assembleForces(double t) const {
    std::vector<double> Q(totalDynNv_, 0.0);

    // Clear and apply all force elements
    for (auto& body : bodies_)
        body->clearForces();
    for (auto& force : forces_)
        force->apply(t);
    // Apply constraint-level damping (e.g. revolute joint damping)
    for (auto& c : constraints_)
        c->applyDamping();

    // Collect generalized forces for dynamic bodies only
    int nDynF = (int)dynBodyIndices_.size();
    #pragma omp parallel for schedule(static)
    for (int di = 0; di < nDynF; di++) {
        int b = dynBodyIndices_[di];
        auto fvec = bodies_[b]->computeForces(gravity_);
        int off = dynVOffsets_[di];
        int n = state_.nvPerBody[b];
        for (int i = 0; i < n && i < (int)fvec.size(); i++)
            Q[off + i] = fvec[i];
    }
    return Q;
}

std::vector<double> MultibodySystem::assembleGamma() const {
    int nc = numConstraintEquations();
    if (nc == 0) return {};

    std::vector<double> gamma;
    gamma.reserve(nc);
    for (auto& c : constraints_) {
        auto g = c->getGamma();
        gamma.insert(gamma.end(), g.begin(), g.end());
    }
    return gamma;
}

int MultibodySystem::numConstraintEquations() const {
    int nc = 0;
    for (auto& c : constraints_)
        nc += c->numEquations();
    return nc;
}

// ---- Solve ----

SolverResult MultibodySystem::solveAccelerations(double t) {
    // NOTE: body state must already be set by the caller (e.g., derivative function).
    auto M = assembleMassMatrix();
    auto Cq = assembleJacobian();
    auto Q = assembleForces(t);
    auto gamma = assembleGamma();

    return solver_->solve(M, Cq, Q, gamma, {});
}

KKTResult MultibodySystem::solveKKTAtState(double t, StateVector& st) {
    // 1. Sync bodies from the given state
    auto ptrs = bodyPtrs();
    for (int b = 0; b < (int)ptrs.size(); b++)
        st.copyToBody(b, ptrs[b]);

    // 2. Solve KKT at this state: [M Cq^T; Cq 0] * [a; λ] = [Q; γ]
    auto solverResult = solveAccelerations(t);

    // 3. Map dynamic-only accelerations back to full-state v-space
    KKTResult result;
    result.accel.resize(st.totalNv, 0.0);
    for (int di = 0; di < (int)dynBodyIndices_.size(); di++) {
        int b      = dynBodyIndices_[di];
        int srcOff = dynVOffsets_[di];
        int dstOff = st.vOffsets[b];
        int n      = st.nvPerBody[b];
        for (int j = 0; j < n; j++) {
            double acc = (srcOff + j < (int)solverResult.x.size())
                       ? solverResult.x[srcOff + j] : 0.0;
            result.accel[dstOff + j] = acc;
        }
    }

    // 4. Lagrange multipliers (packed after accelerations in solver result)
    int nc = numConstraintEquations();
    result.lambda.resize(nc, 0.0);
    for (int i = 0; i < nc && (totalDynNv_ + i) < (int)solverResult.x.size(); i++)
        result.lambda[i] = solverResult.x[totalDynNv_ + i];

    return result;
}

void MultibodySystem::recomputeHHTAPrev(double t) {
    // Solve KKT at the CURRENT body state (after post-projection).
    // The result is injected into HHTAlpha::aPrev_ via setAPrev(), preserving
    // Newmark continuity without the energy injection from invalidateCache().
    auto* hht = dynamic_cast<HHTAlpha*>(integrator_.get());
    if (!hht) return;

    // Bodies are already at the projected state (syncStateToBodie was called by caller).
    auto solverResult = solveAccelerations(t);

    // Map to full-state accelerations
    std::vector<double> a_proj(state_.totalNv, 0.0);
    for (int di = 0; di < (int)dynBodyIndices_.size(); di++) {
        int b      = dynBodyIndices_[di];
        int srcOff = dynVOffsets_[di];
        int dstOff = state_.vOffsets[b];
        int n      = state_.nvPerBody[b];
        for (int j = 0; j < n; j++) {
            a_proj[dstOff + j] = (srcOff + j < (int)solverResult.x.size())
                                ? solverResult.x[srcOff + j] : 0.0;
        }
    }
    hht->setAPrev(a_proj);
}

KKTResult MultibodySystem::solveKKTAtHHTState(
    double t_alpha, StateVector& s_alpha, StateVector& s_np1)
{
    // HHT-DAE formulation (Negrut et al. 2007):
    //   M(q_α) * a + Cq(q_α)^T * λ = Q(t_α, q_α, v_α)   [forces at α-state]
    //   Cq(q_α) * a                 = γ(q_{n+1}, v_{n+1}) [γ (Baumgarte) at n+1]
    //
    // Mass matrix, forces, and constraint Jacobian all at α-state.
    // Only the Baumgarte feedback terms in γ are evaluated at n+1 state,
    // which gives the correct time-level for position/velocity stabilization.

    auto ptrs = bodyPtrs();

    // 1. Sync to α-state → assemble M, Q, and Cq
    for (int b = 0; b < (int)ptrs.size(); b++)
        s_alpha.copyToBody(b, ptrs[b]);
    auto M   = assembleMassMatrix();
    auto Q   = assembleForces(t_alpha);
    auto Cq  = assembleJacobian();

    // 2. Sync to n+1 state → assemble γ (convective + Baumgarte at n+1)
    for (int b = 0; b < (int)ptrs.size(); b++)
        s_np1.copyToBody(b, ptrs[b]);
    auto gamma = assembleGamma();

    // 3. Solve [M Cq^T; Cq 0] * [a; λ] = [Q; γ]
    auto solverResult = solver_->solve(M, Cq, Q, gamma, {});

    // 4. Map back to full-state
    KKTResult result;
    result.accel.resize(s_alpha.totalNv, 0.0);
    for (int di = 0; di < (int)dynBodyIndices_.size(); di++) {
        int b      = dynBodyIndices_[di];
        int srcOff = dynVOffsets_[di];
        int dstOff = s_alpha.vOffsets[b];
        int n      = s_alpha.nvPerBody[b];
        for (int j = 0; j < n; j++) {
            result.accel[dstOff + j] = (srcOff + j < (int)solverResult.x.size())
                                     ? solverResult.x[srcOff + j] : 0.0;
        }
    }
    int nc = numConstraintEquations();
    result.lambda.resize(nc, 0.0);
    for (int i = 0; i < nc && (totalDynNv_ + i) < (int)solverResult.x.size(); i++)
        result.lambda[i] = solverResult.x[totalDynNv_ + i];

    return result;
}

DerivativeFunction MultibodySystem::createDerivativeFunction() {
    return [this](double t, StateVector& st) -> StateVector {
        // Set bodies from the incoming state (full state, all bodies)
        auto ptrs = bodyPtrs();
        for (int b = 0; b < (int)ptrs.size(); b++)
            st.copyToBody(b, ptrs[b]);

        // Process contacts if manager set
        if (contactManager_) {
            contactManager_->processContacts(bodies_, 0.001);
        }

        // Solve accelerations (dynamic-only M, Cq, Q)
        auto result = solveAccelerations(t);

        // Build derivative state (full state size, all bodies)
        StateVector dState = st.clone();
        dState.time = 0.0;

        // q̇ = velocity mapping (all bodies, including ground)
        auto qdot = st.computeQDot(ptrs);
        dState.q = qdot;

        // Zero all v-derivatives first
        int nv = st.totalNv;
        for (int i = 0; i < nv; i++)
            dState.v[i] = 0.0;

        // Map dynamic-only solver result back to full state offsets
        for (int di = 0; di < (int)dynBodyIndices_.size(); di++) {
            int b = dynBodyIndices_[di];
            int srcOff = dynVOffsets_[di];
            int dstOff = st.vOffsets[b];
            int n = st.nvPerBody[b];
            for (int j = 0; j < n; j++) {
                double acc = (srcOff + j < (int)result.x.size()) ? result.x[srcOff + j] : 0.0;
                dState.v[dstOff + j] = acc;
            }
        }

        // Zero out derivatives for static/kinematic bodies
        for (int b = 0; b < (int)ptrs.size(); b++) {
            if (!ptrs[b]->isDynamic()) {
                int qoff = st.qOffsets[b];
                int nqb = st.nqPerBody[b];
                int voff = st.vOffsets[b];
                int nvb = st.nvPerBody[b];
                for (int i = 0; i < nqb; i++) dState.q[qoff + i] = 0.0;
                for (int i = 0; i < nvb; i++) dState.v[voff + i] = 0.0;
            }
        }

        // Store accelerations in state
        for (int i = 0; i < nv; i++)
            st.a[i] = dState.v[i];

        // Lambda (solver result is [dynAccels..., lambda...])
        int nc = numConstraintEquations();
        for (int i = 0; i < nc && (totalDynNv_ + i) < (int)result.x.size(); i++) {
            if (i < (int)dState.lambda.size())
                dState.lambda[i] = result.x[totalDynNv_ + i];
        }

        return dState;
    };
}

StepResult MultibodySystem::step(double dt) {
    if (!initialized_) initialize();

    auto derivFunc = createDerivativeFunction();

    // Correct HHT-DAE: give HHTAlpha a two-state KKT solver.
    // Forces M,Q and Cq at α-state; only γ (Baumgarte feedback) at n+1 state.
    // This is the Negrut 2007 formulation — no post-projection needed.
    if (auto* hht = dynamic_cast<HHTAlpha*>(integrator_.get())) {
        hht->setKKTSolver([this](double t_alpha,
                                 StateVector& s_alpha,
                                 StateVector& s_np1) -> KKTResult {
            return solveKKTAtHHTState(t_alpha, s_alpha, s_np1);
        });
    }

    // Adaptive sub-stepping loop — matches TS step() exactly.
    // Handles both maxStep capping AND integrator rejections.
    double maxStep = 0.0;
    if (integrator_) {
        auto cfg = integrator_->getConfig();
        maxStep = cfg.maxStep;
    }

    double remaining = dt;
    double currentDt = (maxStep > 0) ? std::min(dt, maxStep) : dt;
    const int maxRetries = 20;
    const int maxSubSteps = 500;
    int retries = 0;
    int totalSteps = 0;
    StepResult result;

    while (remaining > 1e-14 && retries < maxRetries && totalSteps < maxSubSteps) {
        currentDt = std::min(currentDt, remaining);
        result = integrator_->step(state_.time, state_, currentDt, derivFunc);

        if (result.accepted) {
            state_ = result.state;
            state_.normalizeQuaternions();
            remaining -= result.dt;
            // Cap nextDt by maxStep
            double next = (result.nextDt > 0) ? result.nextDt : remaining;
            currentDt = (maxStep > 0) ? std::min(next, maxStep) : next;
            totalSteps++;

            // NaN guard
            if (!std::isfinite(state_.q[0]) || !std::isfinite(state_.v[0]))
                break;
        } else {
            // Step rejected — use smaller dt
            double next = (result.nextDt > 0) ? result.nextDt : currentDt * 0.5;
            currentDt = (maxStep > 0) ? std::min(next, maxStep) : next;
            retries++;
        }
    }

    // No post-projection for HHT: projection changes q while keeping v fixed,
    // creating a v-q inconsistency that dissipates energy systematically
    // (especially problematic for chaotic systems like double pendulum).
    // The 0.56J energy "error" from alpha=-0.2 is intrinsic HHT dissipation by design.
    // Explicit integrators: Baumgarte in f() handles constraint drift continuously.

    syncStateToBodie();
    return result;
}

SimStats MultibodySystem::simulate(
    double tf, double dt,
    std::function<void(double, const StateVector&)> callback) {

    if (!initialized_) initialize();

    auto startWall = std::chrono::high_resolution_clock::now();
    SimStats stats;
    int stepCount = 0;

    // Match TS: call step() for each dt interval.
    // step() handles sub-stepping + rejection + projection internally.
    while (state_.time < tf - 1e-14) {
        double h = std::min(dt, tf - state_.time);
        auto result = step(h);
        stepCount++;

        if (callback) callback(state_.time, state_);
    }

    stats.steps = stepCount;
    stats.funcEvals = stepCount;  // approximate
    stats.rejectedSteps = 0;
    if (contactManager_) stats.contactCount = contactManager_->getContactCount();

    auto endWall = std::chrono::high_resolution_clock::now();
    stats.wallTime = std::chrono::duration<double>(endWall - startWall).count();

    return stats;
}

// ---- Constraint projection ----

void MultibodySystem::projectConstraintsPosition() {
    int nc = numConstraintEquations();
    if (nc == 0) return;

    const int maxIter = 5;
    const double tol = 1e-10;

    for (int iter = 0; iter < maxIter; iter++) {
        syncStateToBodie();

        // Gather position violation
        std::vector<double> C;
        C.reserve(nc);
        for (auto& c : constraints_) {
            auto viol = c->computeViolation();
            C.insert(C.end(), viol.position.begin(), viol.position.end());
        }

        double maxViol = 0;
        for (auto v : C) maxViol = std::max(maxViol, std::abs(v));
        if (maxViol < tol) break;

        // Dynamic-only Cq and block-diagonal Minv
        auto Cq = assembleJacobian();  // nc × totalDynNv_

        MatrixN Minv(totalDynNv_, totalDynNv_);
        int nDynP = (int)dynBodyIndices_.size();
        #pragma omp parallel for schedule(static)
        for (int di = 0; di < nDynP; di++) {
            int b = dynBodyIndices_[di];
            int off = dynVOffsets_[di];
            int n = state_.nvPerBody[b];
            auto block = bodies_[b]->computeMassBlock();
            MatrixN Mb(n, n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    Mb.set(i, j, block[i * n + j]);
            auto Mbinv = Mb.inverse();
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    Minv.set(off + i, off + j, Mbinv.get(i, j));
        }

        auto CqT    = Cq.transpose();
        auto CqMinv = Cq.multiply(Minv);
        auto S      = CqMinv.multiply(CqT);

        for (int i = 0; i < nc; i++)
            S.set(i, i, S.get(i, i) + 1e-12);

        auto mu    = S.solve(C);
        auto CqTmu = CqT.multiplyVector(mu);
        auto dv    = Minv.multiplyVector(CqTmu);  // dynamic-only δv

        // Apply position correction: map dynamic-only dv back to bodies
        for (int di = 0; di < (int)dynBodyIndices_.size(); di++) {
            int b = dynBodyIndices_[di];
            int srcOff = dynVOffsets_[di];
            int qoff = state_.qOffsets[b];
            int nq   = state_.nqPerBody[b];
            int n    = state_.nvPerBody[b];

            if (n >= 3 && nq >= 3) {
                state_.q[qoff + 0] -= dv[srcOff + 0];
                state_.q[qoff + 1] -= dv[srcOff + 1];
                state_.q[qoff + 2] -= dv[srcOff + 2];
            }
            if (n >= 6 && nq >= 7) {
                double wx = dv[srcOff + 3];
                double wy = dv[srcOff + 4];
                double wz = dv[srcOff + 5];
                double angle = std::sqrt(wx*wx + wy*wy + wz*wz);
                if (angle > 1e-14) {
                    double ha = angle / 2.0;
                    double s = -std::sin(ha) / angle;
                    Quaternion dq(std::cos(ha), s * wx, s * wy, s * wz);
                    Quaternion cur(state_.q[qoff+3], state_.q[qoff+4],
                                  state_.q[qoff+5], state_.q[qoff+6]);
                    Quaternion corrected = dq.multiply(cur).normalize();
                    state_.q[qoff+3] = corrected.w;
                    state_.q[qoff+4] = corrected.x;
                    state_.q[qoff+5] = corrected.y;
                    state_.q[qoff+6] = corrected.z;
                }
            }
        }
        state_.normalizeQuaternions();
    }
    syncStateToBodie();
}

void MultibodySystem::projectConstraintsVelocity() {
    int nc = numConstraintEquations();
    if (nc == 0) return;

    syncStateToBodie();

    // Compute velocity violation using per-constraint method
    std::vector<double> Cdot(nc, 0.0);
    int row = 0;
    for (auto& c : constraints_) {
        auto cv = c->computeVelocityViolation();
        int neq = c->numEquations();
        for (int i = 0; i < neq && i < (int)cv.size(); i++)
            Cdot[row + i] = cv[i];
        row += neq;
    }

    double maxVel = 0;
    for (auto v : Cdot) maxVel = std::max(maxVel, std::abs(v));
    if (maxVel < 1e-10) return;

    auto Cq = assembleJacobian();  // nc × totalDynNv_

    // Block-diagonal inverse mass (dynamic only)
    MatrixN Minv(totalDynNv_, totalDynNv_);
    int nDynV = (int)dynBodyIndices_.size();
    #pragma omp parallel for schedule(static)
    for (int di = 0; di < nDynV; di++) {
        int b = dynBodyIndices_[di];
        int off = dynVOffsets_[di];
        int n = state_.nvPerBody[b];
        auto block = bodies_[b]->computeMassBlock();
        MatrixN Mb(n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                Mb.set(i, j, block[i * n + j]);
        auto Mbinv = Mb.inverse();
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                Minv.set(off + i, off + j, Mbinv.get(i, j));
    }

    auto CqT    = Cq.transpose();
    auto CqMinv = Cq.multiply(Minv);
    auto S      = CqMinv.multiply(CqT);

    for (int i = 0; i < nc; i++)
        S.set(i, i, S.get(i, i) + 1e-12);

    auto mu    = S.solve(Cdot);
    auto CqTmu = CqT.multiplyVector(mu);
    auto dv    = Minv.multiplyVector(CqTmu);  // dynamic-only

    // Map back to full state
    for (int di = 0; di < (int)dynBodyIndices_.size(); di++) {
        int b = dynBodyIndices_[di];
        int srcOff = dynVOffsets_[di];
        int dstOff = state_.vOffsets[b];
        int n = state_.nvPerBody[b];
        for (int i = 0; i < n; i++)
            state_.v[dstOff + i] -= dv[srcOff + i];
    }
    syncStateToBodie();
}

// ---- Analysis ----

AnalysisResult MultibodySystem::analyze() const {
    AnalysisResult r;
    int nBodies = (int)bodies_.size();
    double totalKE = 0, totalPE = 0;
    double lmx = 0, lmy = 0, lmz = 0;
    double amx = 0, amy = 0, amz = 0;

    #pragma omp parallel for reduction(+:totalKE,totalPE,lmx,lmy,lmz,amx,amy,amz) schedule(static)
    for (int bi = 0; bi < nBodies; bi++) {
        auto& b = bodies_[bi];
        totalKE += b->computeKineticEnergy();
        totalPE += b->computePotentialEnergy(gravity_);
        if (b->isDynamic()) {
            double m = b->getMass();
            lmx += b->velocity.x * m;
            lmy += b->velocity.y * m;
            lmz += b->velocity.z * m;
            auto* rb = dynamic_cast<RigidBody*>(b.get());
            if (rb) {
                Mat3 Iw = rb->getInertiaWorld();
                amx += Iw.get(0,0)*b->angularVelocity.x + Iw.get(0,1)*b->angularVelocity.y + Iw.get(0,2)*b->angularVelocity.z
                      + b->position.y*(b->velocity.z*m) - b->position.z*(b->velocity.y*m);
                amy += Iw.get(1,0)*b->angularVelocity.x + Iw.get(1,1)*b->angularVelocity.y + Iw.get(1,2)*b->angularVelocity.z
                      + b->position.z*(b->velocity.x*m) - b->position.x*(b->velocity.z*m);
                amz += Iw.get(2,0)*b->angularVelocity.x + Iw.get(2,1)*b->angularVelocity.y + Iw.get(2,2)*b->angularVelocity.z
                      + b->position.x*(b->velocity.y*m) - b->position.y*(b->velocity.x*m);
            }
        }
    }
    r.kineticEnergy = totalKE;
    r.potentialEnergy = totalPE;
    r.linearMomentum = Vec3(lmx, lmy, lmz);
    r.angularMomentum = Vec3(amx, amy, amz);
    r.totalEnergy = totalKE + totalPE;

    // Constraint violation
    for (auto& c : constraints_) {
        auto viol = c->computeViolation();
        for (auto v : viol.position)
            r.constraintViolation = std::max(r.constraintViolation, std::abs(v));
        for (auto v : viol.velocity)
            r.velocityViolation = std::max(r.velocityViolation, std::abs(v));
    }
    return r;
}

// ---- Lookup ----

Body* MultibodySystem::getBodyById(int id) const {
    for (auto& b : bodies_)
        if (b->id == id) return b.get();
    return nullptr;
}

Body* MultibodySystem::getBodyByName(const std::string& name) const {
    for (auto& b : bodies_)
        if (b->name == name) return b.get();
    return nullptr;
}

// ---- Helpers ----

std::vector<Body*> MultibodySystem::bodyPtrs() const {
    std::vector<Body*> ptrs;
    ptrs.reserve(bodies_.size());
    for (auto& b : bodies_) ptrs.push_back(b.get());
    return ptrs;
}

void MultibodySystem::syncStateToBodie() const {
    auto ptrs = const_cast<MultibodySystem*>(this)->bodyPtrs();
    for (int b = 0; b < (int)ptrs.size(); b++)
        state_.copyToBody(b, ptrs[b]);
}

void MultibodySystem::syncBodiestoState() {
    auto ptrs = bodyPtrs();
    for (int b = 0; b < (int)ptrs.size(); b++)
        state_.copyFromBody(b, ptrs[b]);
}

} // namespace mb
