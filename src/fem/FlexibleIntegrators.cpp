#include "mb/fem/FlexibleIntegrators.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <map>

namespace mb {

// ═══════════════════════════════════════════════════════════════
//  FlexibleBodyIntegrator (Explicit RK4)
// ═══════════════════════════════════════════════════════════════

FlexibleBodyIntegrator::FlexibleBodyIntegrator(FlexibleBody& body) : body_(body) {}

const std::vector<double>& FlexibleBodyIntegrator::ensureMassInverse() {
    if (!MInvDiag_.empty()) return MInvDiag_;

    int n = body_.numDof;
    const auto& M = body_.getGlobalMassMatrix();
    MInvDiag_.resize(n);

    for (int i = 0; i < n; i++) {
        double mii = M[i*n+i];
        MInvDiag_[i] = (mii > 1e-30 && !body_.isDofFixed(i)) ? 1.0/mii : 0;
    }

    return MInvDiag_;
}

std::vector<double> FlexibleBodyIntegrator::computeAccelerations() {
    const auto& Minv = ensureMassInverse();
    auto Q = body_.computeTotalForces();
    int n = body_.numDof;
    std::vector<double> a(n);
    for (int i = 0; i < n; i++) a[i] = Minv[i] * Q[i];
    return a;
}

FlexStepResult FlexibleBodyIntegrator::step(double dt) {
    int n = body_.numDof;
    auto q0 = body_.getFlexQ();
    auto v0 = body_.getFlexQd();

    // k1
    auto a1 = computeAccelerations();

    // k2
    std::vector<double> q2(n), v2(n);
    for (int i = 0; i < n; i++) {
        q2[i] = q0[i] + 0.5*dt*v0[i];
        v2[i] = v0[i] + 0.5*dt*a1[i];
    }
    body_.setFlexQ(q2); body_.setFlexQd(v2);
    auto a2 = computeAccelerations();

    // k3
    std::vector<double> q3(n), v3(n);
    for (int i = 0; i < n; i++) {
        q3[i] = q0[i] + 0.5*dt*v2[i];
        v3[i] = v0[i] + 0.5*dt*a2[i];
    }
    body_.setFlexQ(q3); body_.setFlexQd(v3);
    auto a3 = computeAccelerations();

    // k4
    std::vector<double> q4(n), v4(n);
    for (int i = 0; i < n; i++) {
        q4[i] = q0[i] + dt*v3[i];
        v4[i] = v0[i] + dt*a3[i];
    }
    body_.setFlexQ(q4); body_.setFlexQd(v4);
    auto a4 = computeAccelerations();

    // Combine
    std::vector<double> qNew(n), vNew(n);
    for (int i = 0; i < n; i++) {
        qNew[i] = q0[i] + (dt/6.0)*(v0[i] + 2*v2[i] + 2*v3[i] + v4[i]);
        vNew[i] = v0[i] + (dt/6.0)*(a1[i] + 2*a2[i] + 2*a3[i] + a4[i]);
    }
    body_.setFlexQ(qNew); body_.setFlexQd(vNew);

    double SE = body_.computeStrainEnergy();
    double KE = body_.computeKineticEnergy();
    double PE = body_.computePotentialEnergy(body_.gravity) - SE;
    return {body_.getMaxDisplacement(), SE, KE, PE};
}

// ═══════════════════════════════════════════════════════════════
//  FlexDOPRI45 (Dormand-Prince 4(5) Adaptive)
// ═══════════════════════════════════════════════════════════════

FlexDOPRI45::FlexDOPRI45(FlexibleBody& body) : body_(body) {}

const std::vector<double>& FlexDOPRI45::ensureMassInverse() {
    if (!MInvDiag_.empty()) return MInvDiag_;
    int n = body_.numDof;
    const auto& M = body_.getGlobalMassMatrix();
    MInvDiag_.resize(n);
    for (int i = 0; i < n; i++) {
        double mii = M[i*n+i];
        MInvDiag_[i] = (mii > 1e-30 && !body_.isDofFixed(i)) ? 1.0/mii : 0;
    }
    return MInvDiag_;
}

std::vector<double> FlexDOPRI45::computeAccelerations() {
    const auto& Minv = ensureMassInverse();
    auto Q = body_.computeTotalForces();
    int n = body_.numDof;
    std::vector<double> a(n);
    for (int i = 0; i < n; i++) a[i] = Minv[i] * Q[i];
    return a;
}

bool FlexDOPRI45::tryStep(double h) {
    // DOPRI5(4) Butcher tableau
    constexpr double a21 = 1.0/5;
    constexpr double a31 = 3.0/40,      a32 = 9.0/40;
    constexpr double a41 = 44.0/45,     a42 = -56.0/15,      a43 = 32.0/9;
    constexpr double a51 = 19372.0/6561,a52 = -25360.0/2187, a53 = 64448.0/6561, a54 = -212.0/729;
    constexpr double a61 = 9017.0/3168, a62 = -355.0/33,     a63 = 46732.0/5247, a64 = 49.0/176,    a65 = -5103.0/18656;
    constexpr double b1  = 35.0/384,    b3  = 500.0/1113,    b4  = 125.0/192,    b5  = -2187.0/6784, b6  = 11.0/84;
    constexpr double e1  = 71.0/57600,  e3  = -71.0/16695,   e4  = 71.0/1920,
                     e5  = -17253.0/339200, e6 = 22.0/525,    e7  = -1.0/40;

    const int n = body_.numDof;

    // Save initial state
    auto q0 = body_.getFlexQ();
    auto v0 = body_.getFlexQd();

    // ── The combined ODE is  y = [q, v],  dy/dt = [v, M^{-1} Q(q,v)]
    // Each stage computes  k_i = f(y_i) = [v_i, a_i]
    // where v_i is the velocity at the stage point and a_i the acceleration.

    // Stage 1 (FSAL)
    std::vector<double> kv1 = hasFSAL_ ? cachedA_ : computeAccelerations();
    // kq1 = v0 (velocity derivative of q at initial state)

    // Temporary buffers
    std::vector<double> qs(n), vs(n);

    // Stage 2  (c2 = 1/5)
    for (int i = 0; i < n; i++) {
        qs[i] = q0[i] + h * a21 * v0[i];
        vs[i] = v0[i] + h * a21 * kv1[i];
    }
    body_.setFlexQ(qs); body_.setFlexQd(vs);
    std::vector<double> kv2 = computeAccelerations();
    // kq2 = vs  (but we need the values frozen, so copy)
    std::vector<double> kq2 = vs;

    // Stage 3  (c3 = 3/10)
    for (int i = 0; i < n; i++) {
        qs[i] = q0[i] + h * (a31*v0[i]  + a32*kq2[i]);
        vs[i] = v0[i] + h * (a31*kv1[i] + a32*kv2[i]);
    }
    body_.setFlexQ(qs); body_.setFlexQd(vs);
    std::vector<double> kv3 = computeAccelerations();
    std::vector<double> kq3 = vs;

    // Stage 4  (c4 = 4/5)
    for (int i = 0; i < n; i++) {
        qs[i] = q0[i] + h * (a41*v0[i]  + a42*kq2[i] + a43*kq3[i]);
        vs[i] = v0[i] + h * (a41*kv1[i] + a42*kv2[i] + a43*kv3[i]);
    }
    body_.setFlexQ(qs); body_.setFlexQd(vs);
    std::vector<double> kv4 = computeAccelerations();
    std::vector<double> kq4 = vs;

    // Stage 5  (c5 = 8/9)
    for (int i = 0; i < n; i++) {
        qs[i] = q0[i] + h * (a51*v0[i]  + a52*kq2[i] + a53*kq3[i] + a54*kq4[i]);
        vs[i] = v0[i] + h * (a51*kv1[i] + a52*kv2[i] + a53*kv3[i] + a54*kv4[i]);
    }
    body_.setFlexQ(qs); body_.setFlexQd(vs);
    std::vector<double> kv5 = computeAccelerations();
    std::vector<double> kq5 = vs;

    // Stage 6  (c6 = 1)
    for (int i = 0; i < n; i++) {
        qs[i] = q0[i] + h * (a61*v0[i]  + a62*kq2[i] + a63*kq3[i] + a64*kq4[i] + a65*kq5[i]);
        vs[i] = v0[i] + h * (a61*kv1[i] + a62*kv2[i] + a63*kv3[i] + a64*kv4[i] + a65*kv5[i]);
    }
    body_.setFlexQ(qs); body_.setFlexQd(vs);
    std::vector<double> kv6 = computeAccelerations();
    std::vector<double> kq6 = vs;

    // ── 5th-order solution ──
    std::vector<double> q5(n), v5(n);
    for (int i = 0; i < n; i++) {
        q5[i] = q0[i] + h * (b1*v0[i]  + b3*kq3[i] + b4*kq4[i] + b5*kq5[i] + b6*kq6[i]);
        v5[i] = v0[i] + h * (b1*kv1[i] + b3*kv3[i] + b4*kv4[i] + b5*kv5[i] + b6*kv6[i]);
    }
    body_.setFlexQ(q5); body_.setFlexQd(v5);

    // Stage 7 (FSAL — reused as k1 of next step)
    std::vector<double> kv7 = computeAccelerations();

    // ── Error estimate ──
    double errSq = 0;
    int nActive = 0;
    for (int i = 0; i < n; i++) {
        if (body_.isDofFixed(i)) continue;
        nActive++;
        double eq = h * (e1*v0[i]  + e3*kq3[i] + e4*kq4[i] + e5*kq5[i] + e6*kq6[i] + e7*v5[i]);
        double ev = h * (e1*kv1[i] + e3*kv3[i] + e4*kv4[i] + e5*kv5[i] + e6*kv6[i] + e7*kv7[i]);
        double sq = absTol + relTol * std::max(std::abs(q0[i]), std::abs(q5[i]));
        double sv = absTol + relTol * std::max(std::abs(v0[i]), std::abs(v5[i]));
        errSq += (eq/sq)*(eq/sq) + (ev/sv)*(ev/sv);
    }
    double err = std::sqrt(errSq / std::max(1, 2*nActive));

    constexpr double safety = 0.9;
    if (err <= 1.0) {
        cachedA_ = kv7;
        hasFSAL_ = true;
        totalSteps++;
        double factor = (err > 1e-10) ? safety * std::pow(1.0/err, 0.2) : 5.0;
        dtCurrent = std::clamp(h * std::min(factor, 5.0), minStep, maxStep);
        return true;
    } else {
        body_.setFlexQ(q0); body_.setFlexQd(v0);
        hasFSAL_ = false;
        totalRejects++;
        double factor = safety * std::pow(1.0/err, 0.25);
        dtCurrent = std::max(minStep, h * std::max(factor, 0.1));
        return false;
    }
}

FlexStepResult FlexDOPRI45::step(double dtTarget) {
    double tRemain = dtTarget;
    while (tRemain > 1e-14) {
        double h = std::min(dtCurrent, tRemain);
        if (tryStep(h)) {
            tRemain -= h;
        }
        // If rejected, dtCurrent was already reduced; loop retries
    }
    double SE = body_.computeStrainEnergy();
    double KE = body_.computeKineticEnergy();
    double PE = body_.computePotentialEnergy(body_.gravity) - SE;
    return {body_.getMaxDisplacement(), SE, KE, PE};
}

const std::vector<double>& ImplicitFlexIntegrator::ensureMassInverse() {
    if (!MInvDiag_.empty()) return MInvDiag_;
    int n = body_.numDof;
    const auto& M = body_.getGlobalMassMatrix();
    MInvDiag_.resize(n);
    for (int i = 0; i < n; i++) {
        double mii = M[i*n+i];
        MInvDiag_[i] = (mii > 1e-30 && !body_.isDofFixed(i)) ? 1.0/mii : 0;
    }
    return MInvDiag_;
}

std::vector<int> ImplicitFlexIntegrator::getFreeDofMap() const {
    int n = body_.numDof;
    std::vector<int> map;
    for (int i = 0; i < n; i++)
        if (!body_.isDofFixed(i)) map.push_back(i);
    return map;
}

std::vector<double> ImplicitFlexIntegrator::solveDenseLU(
    std::vector<double>& A, const std::vector<double>& b, int n)
{
    std::vector<int> piv(n);
    for (int i = 0; i < n; i++) piv[i] = i;

    for (int k = 0; k < n; k++) {
        double maxVal = std::abs(A[piv[k]*n+k]);
        int maxRow = k;
        for (int i = k+1; i < n; i++) {
            double val = std::abs(A[piv[i]*n+k]);
            if (val > maxVal) { maxVal = val; maxRow = i; }
        }
        if (maxRow != k) std::swap(piv[k], piv[maxRow]);

        double pkk = A[piv[k]*n+k];
        if (std::abs(pkk) < 1e-30) continue;

        for (int i = k+1; i < n; i++) {
            double factor = A[piv[i]*n+k] / pkk;
            A[piv[i]*n+k] = factor;
            for (int j = k+1; j < n; j++)
                A[piv[i]*n+j] -= factor * A[piv[k]*n+j];
        }
    }

    std::vector<double> y(n);
    for (int i = 0; i < n; i++) {
        y[i] = b[piv[i]];
        for (int j = 0; j < i; j++) y[i] -= A[piv[i]*n+j] * y[j];
    }
    std::vector<double> x(n);
    for (int i = n-1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i+1; j < n; j++) x[i] -= A[piv[i]*n+j] * x[j];
        double d = A[piv[i]*n+i];
        if (std::abs(d) > 1e-30) x[i] /= d; else x[i] = 0;
    }
    return x;
}

FlexStepResult ImplicitFlexIntegrator::step(double dt) {
    int n = body_.numDof;
    double alphaHht = std::clamp(hhtAlpha, -1.0/3.0, 0.0);
    double beta = alphaHht == 0 ? 0.25 : 0.25*(1-alphaHht)*(1-alphaHht);
    double gamma = alphaHht == 0 ? 0.5 : 0.5 - alphaHht;
    double h = dt, h2 = h*h;
    double hhtForceScale = 1.0 + alphaHht;

    auto freeMap = getFreeDofMap();
    int nf = (int)freeMap.size();

    auto q0 = body_.getFlexQ();
    auto v0 = body_.getFlexQd();

    // Set current state and compute Q0
    body_.setFlexQ(q0); body_.setFlexQd(v0);
    auto Q0 = body_.computeTotalForces();

    const auto& M = body_.getGlobalMassMatrix();

    // Initial acceleration
    if (aPrev_.empty()) {
        const auto& Minv = ensureMassInverse();
        aPrev_.resize(n);
        for (int i = 0; i < n; i++) aPrev_[i] = Minv[i] * Q0[i];
    }

    // Predictors
    std::vector<double> qPred(n), vPred(n);
    for (int i = 0; i < n; i++) {
        qPred[i] = q0[i] + h*v0[i] + (0.5-beta)*h2*aPrev_[i];
        vPred[i] = v0[i] + (1.0-gamma)*h*aPrev_[i];
    }

    std::vector<double> a = aPrev_;
    std::vector<double> qCurr(n), vCurr(n);

    for (int newtonIter = 0; newtonIter < maxNewtonIter; newtonIter++) {
        for (int i = 0; i < n; i++) {
            qCurr[i] = qPred[i] + beta*h2*a[i];
            vCurr[i] = vPred[i] + gamma*h*a[i];
        }

        body_.setFlexQ(qCurr); body_.setFlexQd(vCurr);
        auto Q = body_.computeTotalForces();

        // Residual
        std::vector<double> R(nf);
        for (int ii = 0; ii < nf; ii++) {
            int i = freeMap[ii];
            double Ma_i = 0;
            for (int jj = 0; jj < nf; jj++)
                Ma_i += M[i*n+freeMap[jj]] * a[freeMap[jj]];
            double Qeff = hhtForceScale*Q[i] - alphaHht*Q0[i];
            R[ii] = Ma_i - Qeff;
        }

        double rNorm = 0;
        for (int ii = 0; ii < nf; ii++) rNorm += R[ii]*R[ii];
        rNorm = std::sqrt(rNorm);
        if (rNorm < newtonTol) break;

        // Build tangent via central FD
        std::vector<double> S(nf*nf);
        double dampFactor = 1.0 + hhtForceScale*gamma*h*body_.dampingAlpha;
        for (int ii = 0; ii < nf; ii++)
            for (int jj = 0; jj < nf; jj++)
                S[ii*nf+jj] = dampFactor * M[freeMap[ii]*n+freeMap[jj]];

        for (int jj = 0; jj < nf; jj++) {
            int j = freeMap[jj];
            double eps_j = fdEps * std::max(std::abs(qCurr[j]), 1.0);

            auto qPlus = qCurr; qPlus[j] += eps_j;
            body_.setFlexQ(qPlus); body_.setFlexQd(vCurr);
            auto QPlus = body_.computeTotalForces();

            auto qMinus = qCurr; qMinus[j] -= eps_j;
            body_.setFlexQ(qMinus); body_.setFlexQd(vCurr);
            auto QMinus = body_.computeTotalForces();

            for (int ii = 0; ii < nf; ii++) {
                double dQdq = (QPlus[freeMap[ii]] - QMinus[freeMap[ii]]) / (2*eps_j);
                S[ii*nf+jj] -= hhtForceScale * beta * h2 * dQdq;
            }
        }

        body_.setFlexQ(qCurr); body_.setFlexQd(vCurr);

        // Solve
        std::vector<double> negR(nf);
        for (int ii = 0; ii < nf; ii++) negR[ii] = -R[ii];
        auto da = solveDenseLU(S, negR, nf);

        // Update with line search
        double alpha = 1.0;
        for (int ii = 0; ii < nf; ii++)
            a[freeMap[ii]] += alpha * da[ii];
    }

    // Final state
    for (int i = 0; i < n; i++) {
        qCurr[i] = qPred[i] + beta*h2*a[i];
        vCurr[i] = vPred[i] + gamma*h*a[i];
    }
    body_.setFlexQ(qCurr); body_.setFlexQd(vCurr);
    aPrev_ = a;
    stepCount_++;

    double SE = body_.computeStrainEnergy();
    double KE = body_.computeKineticEnergy();
    double PE = body_.computePotentialEnergy(body_.gravity) - SE;
    return {body_.getMaxDisplacement(), SE, KE, PE};
}

// ═══════════════════════════════════════════════════════════════
//  Static Equilibrium
// ═══════════════════════════════════════════════════════════════

static std::vector<double> staticSolveLU(std::vector<double>& A, const std::vector<double>& b, int n) {
    std::vector<int> piv(n);
    for (int i = 0; i < n; i++) piv[i] = i;
    for (int k = 0; k < n; k++) {
        double maxVal = std::abs(A[piv[k]*n+k]); int maxRow = k;
        for (int i = k+1; i < n; i++) {
            double val = std::abs(A[piv[i]*n+k]);
            if (val > maxVal) { maxVal = val; maxRow = i; }
        }
        if (maxRow != k) std::swap(piv[k], piv[maxRow]);
        double pkk = A[piv[k]*n+k];
        if (std::abs(pkk) < 1e-30) continue;
        for (int i = k+1; i < n; i++) {
            double f = A[piv[i]*n+k] / pkk;
            A[piv[i]*n+k] = f;
            for (int j = k+1; j < n; j++) A[piv[i]*n+j] -= f*A[piv[k]*n+j];
        }
    }
    std::vector<double> y(n);
    for (int i = 0; i < n; i++) {
        y[i] = b[piv[i]];
        for (int j = 0; j < i; j++) y[i] -= A[piv[i]*n+j]*y[j];
    }
    std::vector<double> x(n);
    for (int i = n-1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i+1; j < n; j++) x[i] -= A[piv[i]*n+j]*x[j];
        double d = A[piv[i]*n+i];
        if (std::abs(d) > 1e-30) x[i] /= d; else x[i] = 0;
    }
    return x;
}

StaticSolveResult solveStaticEquilibrium(FlexibleBody& body, const StaticSolveOptions& opts) {
    int n = body.numDof;
    std::vector<int> freeIdx;
    for (int i = 0; i < n; i++)
        if (!body.isDofFixed(i)) freeIdx.push_back(i);
    int nf = (int)freeIdx.size();

    Vec3 origGravity = body.gravity;
    auto origExt = body.externalForces;
    int totalIter = 0;
    double finalRes = 0;

    for (int loadStep = 1; loadStep <= opts.nLoadSteps; loadStep++) {
        double loadFactor = (double)loadStep / opts.nLoadSteps;
        body.gravity = origGravity * loadFactor;

        if (origExt) {
            body.externalForces = [&origExt, loadFactor](FlexibleBody& b) {
                auto Qext = origExt(b);
                for (auto& v : Qext) v *= loadFactor;
                return Qext;
            };
        }

        for (int iter = 0; iter < opts.maxIter; iter++) {
            totalIter++;
            std::vector<double> zeroVel(n, 0.0);
            body.setFlexQd(zeroVel);
            auto Q = body.computeTotalForces();

            std::vector<double> R(nf);
            for (int ii = 0; ii < nf; ii++) R[ii] = -Q[freeIdx[ii]];

            double rNorm = 0;
            for (int ii = 0; ii < nf; ii++) rNorm += R[ii]*R[ii];
            rNorm = std::sqrt(rNorm);

            if (opts.verbose)
                std::cout << "  iter " << iter << ": |R| = " << rNorm << "\n";

            if (rNorm < opts.tol) { finalRes = rNorm; break; }

            // Build tangent stiffness K via central FD
            auto qSave = body.getFlexQ();
            std::vector<double> K(nf*nf);
            for (int jj = 0; jj < nf; jj++) {
                int j = freeIdx[jj];
                double eps_j = opts.fdEps * std::max(std::abs(qSave[j]), 1.0);

                auto qPlus = qSave; qPlus[j] += eps_j;
                body.setFlexQ(qPlus); body.setFlexQd(zeroVel);
                auto QPlus = body.computeTotalForces();

                auto qMinus = qSave; qMinus[j] -= eps_j;
                body.setFlexQ(qMinus); body.setFlexQd(zeroVel);
                auto QMinus = body.computeTotalForces();

                for (int ii = 0; ii < nf; ii++)
                    K[ii*nf+jj] = -(QPlus[freeIdx[ii]] - QMinus[freeIdx[ii]]) / (2*eps_j);
            }
            body.setFlexQ(qSave);

            // Solve K dq = -R
            std::vector<double> negR(nf);
            for (int ii = 0; ii < nf; ii++) negR[ii] = -R[ii];
            auto dq = staticSolveLU(K, negR, nf);

            // Line search
            double alpha = 1.0;
            for (int ls = 0; ls < 10; ls++) {
                auto qTrial = qSave;
                for (int ii = 0; ii < nf; ii++)
                    qTrial[freeIdx[ii]] += alpha * dq[ii];
                body.setFlexQ(qTrial); body.setFlexQd(zeroVel);
                auto QTrial = body.computeTotalForces();
                double rTrial = 0;
                for (int ii = 0; ii < nf; ii++) {
                    double v = -QTrial[freeIdx[ii]];
                    rTrial += v*v;
                }
                rTrial = std::sqrt(rTrial);
                if (rTrial < rNorm || alpha < 1e-4) break;
                alpha *= 0.5;
            }

            auto qNew = qSave;
            for (int ii = 0; ii < nf; ii++)
                qNew[freeIdx[ii]] += alpha * dq[ii];
            body.setFlexQ(qNew);
            finalRes = rNorm;
        }
    }

    body.externalForces = origExt;
    body.gravity = origGravity;

    return {finalRes < opts.tol, totalIter, finalRes,
            body.getMaxDisplacement(), body.computeStrainEnergy()};
}

// ═══════════════════════════════════════════════════════════════
//  Mesh Generators
// ═══════════════════════════════════════════════════════════════

GmshMesh generateBoxTetMesh(double Lx, double Ly, double Lz,
                            int nx, int ny, int nz)
{
    GmshMesh mesh;
    double dx = Lx/nx, dy = Ly/ny, dz = Lz/nz;

    auto nodeIndex = [&](int ix, int iy, int iz) {
        return ix + (nx+1)*(iy + (ny+1)*iz);
    };

    int nid = 0;
    for (int iz = 0; iz <= nz; iz++)
        for (int iy = 0; iy <= ny; iy++)
            for (int ix = 0; ix <= nx; ix++)
                mesh.nodes.push_back({nid++, ix*dx, iy*dy, iz*dz});

    // Alternating 5-tet Kuhn triangulation: adjacent cells use
    // opposite body diagonals → face-compatible & symmetric mesh.
    int eid = 0;
    for (int iz = 0; iz < nz; iz++)
        for (int iy = 0; iy < ny; iy++)
            for (int ix = 0; ix < nx; ix++) {
                int v[8] = {
                    nodeIndex(ix,iy,iz), nodeIndex(ix+1,iy,iz),
                    nodeIndex(ix+1,iy+1,iz), nodeIndex(ix,iy+1,iz),
                    nodeIndex(ix,iy,iz+1), nodeIndex(ix+1,iy,iz+1),
                    nodeIndex(ix+1,iy+1,iz+1), nodeIndex(ix,iy+1,iz+1)
                };
                int tets[5][4];
                if ((ix + iy + iz) % 2 == 0) {
                    // Type A: body diagonal 0→6
                    tets[0][0]=v[0]; tets[0][1]=v[1]; tets[0][2]=v[2]; tets[0][3]=v[5];
                    tets[1][0]=v[0]; tets[1][1]=v[2]; tets[1][2]=v[3]; tets[1][3]=v[7];
                    tets[2][0]=v[0]; tets[2][1]=v[4]; tets[2][2]=v[5]; tets[2][3]=v[7];
                    tets[3][0]=v[2]; tets[3][1]=v[5]; tets[3][2]=v[6]; tets[3][3]=v[7];
                    tets[4][0]=v[0]; tets[4][1]=v[2]; tets[4][2]=v[5]; tets[4][3]=v[7];
                } else {
                    // Type B: body diagonal 1→7
                    tets[0][0]=v[0]; tets[0][1]=v[1]; tets[0][2]=v[3]; tets[0][3]=v[4];
                    tets[1][0]=v[1]; tets[1][1]=v[2]; tets[1][2]=v[3]; tets[1][3]=v[6];
                    tets[2][0]=v[1]; tets[2][1]=v[4]; tets[2][2]=v[5]; tets[2][3]=v[6];
                    tets[3][0]=v[3]; tets[3][1]=v[4]; tets[3][2]=v[6]; tets[3][3]=v[7];
                    tets[4][0]=v[1]; tets[4][1]=v[3]; tets[4][2]=v[4]; tets[4][3]=v[6];
                }
                for (auto& t : tets)
                    mesh.elements.push_back({eid++, 4, {t[0],t[1],t[2],t[3]}});
            }

    return mesh;
}

GmshMesh generateCylinderTetMesh(double R, double L,
                                  int nR, int nT, int nZ,
                                  double innerRadius)
{
    GmshMesh mesh;
    double dz = L / nZ;
    double Ri = std::clamp(innerRadius, 0.0, R * 0.999999);
    bool isSolid = Ri <= 1e-12;

    std::map<std::string, int> nodeMap;
    int nid = 0;

    auto getOrCreate = [&](int ir, int it, int iz) -> int {
        bool collapse = isSolid && ir == 0;
        std::string key = collapse
            ? "0-0-" + std::to_string(iz)
            : std::to_string(ir) + "-" + std::to_string(it % nT) + "-" + std::to_string(iz);

        auto it2 = nodeMap.find(key);
        if (it2 != nodeMap.end()) return it2->second;

        double x, y, z;
        if (collapse) { x = 0; y = 0; }
        else {
            double r = isSolid ? ((double)ir/nR)*R : Ri + ((double)ir/nR)*(R-Ri);
            double theta = 2.0*M_PI*(it % nT) / nT;
            x = r*std::cos(theta); y = r*std::sin(theta);
        }
        z = -L/2.0 + iz*dz;

        int id = nid++;
        nodeMap[key] = id;
        mesh.nodes.push_back({id, x, y, z});
        return id;
    };

    // Create all nodes
    for (int iz = 0; iz <= nZ; iz++)
        for (int ir = 0; ir <= nR; ir++) {
            if (isSolid && ir == 0) getOrCreate(0, 0, iz);
            else for (int it = 0; it < nT; it++) getOrCreate(ir, it, iz);
        }

    // Create elements
    int eid = 0;
    for (int iz = 0; iz < nZ; iz++)
        for (int ir = 0; ir < nR; ir++)
            for (int it = 0; it < nT; it++) {
                int it1 = (it+1) % nT;
                if (isSolid && ir == 0) {
                    int c0 = getOrCreate(0,0,iz), c1 = getOrCreate(0,0,iz+1);
                    int a0 = getOrCreate(1,it,iz), b0 = getOrCreate(1,it1,iz);
                    int a1 = getOrCreate(1,it,iz+1), b1 = getOrCreate(1,it1,iz+1);
                    mesh.elements.push_back({eid++, 4, {c0,a0,b0,c1}});
                    mesh.elements.push_back({eid++, 4, {a0,b0,c1,a1}});
                    mesh.elements.push_back({eid++, 4, {b0,c1,a1,b1}});
                } else {
                    int v[8] = {
                        getOrCreate(ir,it,iz), getOrCreate(ir+1,it,iz),
                        getOrCreate(ir+1,it1,iz), getOrCreate(ir,it1,iz),
                        getOrCreate(ir,it,iz+1), getOrCreate(ir+1,it,iz+1),
                        getOrCreate(ir+1,it1,iz+1), getOrCreate(ir,it1,iz+1)
                    };
                    int tets[6][4] = {
                        {v[0],v[1],v[2],v[6]}, {v[0],v[1],v[6],v[5]},
                        {v[0],v[4],v[5],v[6]}, {v[0],v[2],v[3],v[6]},
                        {v[0],v[3],v[7],v[6]}, {v[0],v[4],v[6],v[7]}
                    };
                    for (auto& t : tets)
                        mesh.elements.push_back({eid++, 4, {t[0],t[1],t[2],t[3]}});
                }
            }

    return mesh;
}

} // namespace mb
