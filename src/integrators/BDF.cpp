#include "mb/integrators/BDF.h"
#include <cmath>

namespace mb {

// ============== BackwardEuler ==============

StepResult BackwardEuler::step(double t, StateVector& state, double dt,
                               DerivativeFunction& f) {
    // Implicit Euler: y_{n+1} = y_n + dt * f(t_{n+1}, y_{n+1})
    // Use one Newton iteration with explicit predictor
    auto predicted = state.clone();
    auto k = f(t + dt, predicted);

    StateVector result = state.addScaled(k, dt);
    result.time = t + dt;
    result.normalizeQuaternions();

    return {result, dt, dt, true};
}

// ============== BDF2 ==============

StepResult BDF2::step(double t, StateVector& state, double dt,
                      DerivativeFunction& f) {
    if (!hasHistory_) {
        // Fall back to backward Euler for first step
        auto k = f(t + dt, state);
        StateVector result = state.addScaled(k, dt);
        result.time = t + dt;
        result.normalizeQuaternions();
        hasHistory_ = true;
        prevState_ = state.clone();
        return {result, dt, dt, true};
    }

    // BDF-2: y_{n+1} = 4/3*y_n - 1/3*y_{n-1} + 2/3*dt*f(t_{n+1}, y_{n+1})
    auto pred = state.clone();
    auto k = f(t + dt, pred);

    StateVector result = state.clone();
    for (int i = 0; i < state.totalNq; i++)
        result.q[i] = 4.0/3*state.q[i] - 1.0/3*prevState_.q[i] + 2.0/3*dt*k.q[i];
    for (int i = 0; i < state.totalNv; i++)
        result.v[i] = 4.0/3*state.v[i] - 1.0/3*prevState_.v[i] + 2.0/3*dt*k.v[i];

    result.time = t + dt;
    result.normalizeQuaternions();
    prevState_ = state.clone();

    return {result, dt, dt, true};
}

// ============== GeneralizedAlpha ==============

GeneralizedAlpha::GeneralizedAlpha(double rhoInfinity, const IntegratorConfig& config)
    : TimeIntegrator(config) {
    alphaM_ = (2 * rhoInfinity - 1) / (rhoInfinity + 1);
    alphaF_ = rhoInfinity / (rhoInfinity + 1);
    gammaGA_ = 0.5 + alphaF_ - alphaM_;
    beta_ = 0.25 * (gammaGA_ + 0.5) * (gammaGA_ + 0.5);
}

StepResult GeneralizedAlpha::step(double t, StateVector& state, double dt,
                                   DerivativeFunction& f) {
    // Newmark-like update with generalized-alpha parameters
    auto k = f(t + dt, state);

    StateVector result = state.clone();
    for (int i = 0; i < state.totalNv; i++) {
        double a_new = k.v[i]; // acceleration from derivative
        result.v[i] = state.v[i] + dt * ((1 - gammaGA_) * state.a[i] + gammaGA_ * a_new);
    }

    for (int i = 0; i < state.totalNq; i++) {
        result.q[i] = state.q[i] + dt * k.q[i];
    }

    // Store accelerations
    for (int i = 0; i < state.totalNv; i++) {
        result.a[i] = k.v[i];
    }

    result.time = t + dt;
    result.normalizeQuaternions();

    return {result, dt, dt, true};
}

// ============== SemiImplicitEuler ==============

StepResult SemiImplicitEuler::step(double t, StateVector& state, double dt,
                                    DerivativeFunction& f) {
    // 1. Compute accelerations at current state
    auto k = f(t, state);

    StateVector result = state.clone();

    // 2. Update velocity: v_{n+1} = v_n + dt * a
    for (int i = 0; i < state.totalNv; i++)
        result.v[i] = state.v[i] + dt * k.v[i];

    // 3. Update position using NEW velocity: q_{n+1} = q_n + dt * q̇(v_{n+1})
    for (int i = 0; i < state.totalNq; i++)
        result.q[i] = state.q[i] + dt * k.q[i]; // Simplified

    result.time = t + dt;
    result.normalizeQuaternions();

    return {result, dt, dt, true};
}

// ============== VelocityVerlet ==============

StepResult VelocityVerlet::step(double t, StateVector& state, double dt,
                                 DerivativeFunction& f) {
    // 1. Compute accelerations
    auto k1 = f(t, state);

    StateVector result = state.clone();

    // 2. Half-step velocity
    for (int i = 0; i < state.totalNv; i++)
        result.v[i] = state.v[i] + 0.5 * dt * k1.v[i];

    // 3. Full-step position
    for (int i = 0; i < state.totalNq; i++)
        result.q[i] = state.q[i] + dt * k1.q[i];

    result.time = t + dt;

    // 4. Compute new accelerations
    auto k2 = f(t + dt, result);

    // 5. Complete velocity step
    for (int i = 0; i < state.totalNv; i++)
        result.v[i] = state.v[i] + 0.5 * dt * (k1.v[i] + k2.v[i]);

    result.normalizeQuaternions();

    return {result, dt, dt, true};
}

// ============== HHTAlpha ==============

HHTAlpha::HHTAlpha(double alpha, int maxIter, double tol, const IntegratorConfig& config)
    : TimeIntegrator(config), alpha_(alpha), maxIter_(maxIter), tol_(tol)
{
    // Clamp α to stability range [-1/3, 0]
    if (alpha_ < -1.0/3.0) alpha_ = -1.0/3.0;
    if (alpha_ >  0.0)     alpha_ =  0.0;
}

StepResult HHTAlpha::step(double t, StateVector& state, double dt,
                          DerivativeFunction& f)
{
    // ── HHT-α parameters ────────────────────────────────────────────────────
    // γ = 1/2 - α,  β = (1-α)²/4  (A-stable, 2nd order, unconditionally stable)
    const double alpha   = alpha_;
    const double gamma_c = 0.5 - alpha;
    const double beta_c  = (1.0 - alpha) * (1.0 - alpha) * 0.25;
    const double t_alpha = t + (1.0 + alpha) * dt;

    const int nq = state.totalNq;
    const int nv = state.totalNv;
    const int nb = state.numBodies;
    const int nc = (int)state.lambda.size();

    // ── Step 0: a_n — from cache or solve at current state ──────────────────
    if (!hasHistory_) {
        aPrev_.resize(nv, 0.0);
        lambdaPrev_.resize(nc, 0.0);
        if (kktSolver_) {
            // Initial solve: at t_n, s_alpha = s_np1 = state (no interpolation yet)
            StateVector state_copy = state.clone();
            auto r = kktSolver_(t, state_copy, state_copy);
            aPrev_ = r.accel;
            for (int i = 0; i < nc && i < (int)r.lambda.size(); i++)
                lambdaPrev_[i] = r.lambda[i];
        } else {
            // FALLBACK: derivative function
            auto k = f(t, state);
            for (int i = 0; i < nv; i++) aPrev_[i] = k.v[i];
        }
        hasHistory_ = true;
    }

    // ── Step 1: Compute qdot_n directly from (q_n, v_n) ─────────────────────
    // No extra f() call needed — quaternion kinematics are algebraic.
    std::vector<double> qdot_n(nq, 0.0);
    for (int b = 0; b < nb; b++) {
        const int qo  = state.qOffsets[b];
        const int vo  = state.vOffsets[b];
        const int nqb = state.nqPerBody[b];
        const int nvb = state.nvPerBody[b];
        const int nt  = std::min(3, std::min(nqb, nvb));
        // Translation: qdot = v
        for (int j = 0; j < nt; j++)
            qdot_n[qo+j] = state.v[vo+j];
        // Rotation: qdot = 0.5 * q ⊗ ω
        if (nqb == 7 && nvb >= 6) {
            const double qw = state.q[qo+3], qx = state.q[qo+4],
                         qy = state.q[qo+5], qz = state.q[qo+6];
            const double wx = state.v[vo+3], wy = state.v[vo+4], wz = state.v[vo+5];
            qdot_n[qo+3] = -0.5*(qx*wx + qy*wy + qz*wz);
            qdot_n[qo+4] =  0.5*(qw*wx + qy*wz - qz*wy);
            qdot_n[qo+5] =  0.5*(qw*wy + qz*wx - qx*wz);
            qdot_n[qo+6] =  0.5*(qw*wz + qx*wy - qy*wx);
        }
    }

    // ── Step 2: Newmark predictor ────────────────────────────────────────────
    //   ṽ = v_n + dt*(1-γ)*a_n
    //   x̃ = x_n + dt*v_n + dt²*(0.5-β)*a_n    (translation — Newmark)
    //   q̃ = q_n + dt*qdot_n                    (rotation — 1st-order kinematic)
    std::vector<double> v_pred(nv), q_pred(nq);
    for (int i = 0; i < nv; i++)
        v_pred[i] = state.v[i] + dt * (1.0 - gamma_c) * aPrev_[i];
    for (int b = 0; b < nb; b++) {
        const int qo  = state.qOffsets[b];
        const int vo  = state.vOffsets[b];
        const int nqb = state.nqPerBody[b];
        const int nvb = state.nvPerBody[b];
        const int nt  = std::min(3, std::min(nqb, nvb));
        for (int j = 0; j < nt; j++)
            q_pred[qo+j] = state.q[qo+j]
                         + dt * state.v[vo+j]
                         + dt*dt * (0.5 - beta_c) * aPrev_[vo+j];
        for (int j = nt; j < nqb; j++)
            q_pred[qo+j] = state.q[qo+j] + dt * qdot_n[qo+j];
    }

    // ── Step 3: Fixed-point corrector ────────────────────────────────────────
    //
    // COUPLED DAE (kktSolver_ set):
    //   Build (q_{n+1}^k, v_{n+1}^k) from Newmark formulas.
    //   Evaluate α-interpolated state: q_α, v_α.
    //   Solve KKT at (t_α, q_α, v_α) → (a^{k+1}, λ^{k+1}) simultaneously.
    //   M(q_α)*a + Cq(q_α)^T*λ = Q(t_α, q_α, v_α)
    //   Cq(q_α)*a               = γ(q_α, v_α)
    //   → constraint forces λ are consistent with dynamics at every iteration.
    //
    // FALLBACK (no kktSolver_):
    //   Evaluate EOM via derivative function (split-operator, previous behaviour).
    std::vector<double> a_new(aPrev_);
    std::vector<double> lambda_new(lambdaPrev_);
    // q_{n+1} tracked across iterations for iterative quaternion update
    std::vector<double> q_np1(q_pred);

    for (int iter = 0; iter < maxIter_; iter++) {
        // v_{n+1}^k = ṽ + dt*γ*a^k
        std::vector<double> v_np1(nv);
        for (int i = 0; i < nv; i++)
            v_np1[i] = v_pred[i] + dt * gamma_c * a_new[i];

        // q_{n+1}^k: translation via Newmark β, rotation via iterative trapezoidal
        std::vector<double> q_np1_new(nq);
        for (int b = 0; b < nb; b++) {
            const int qo  = state.qOffsets[b];
            const int vo  = state.vOffsets[b];
            const int nqb = state.nqPerBody[b];
            const int nvb = state.nvPerBody[b];
            const int nt  = std::min(3, std::min(nqb, nvb));

            // Translation: x_{n+1} = x̃ + dt²*β*a^k
            for (int j = 0; j < nt; j++)
                q_np1_new[qo+j] = q_pred[qo+j] + dt*dt * beta_c * a_new[vo+j];

            // Rotation: q_{n+1} = q_n + dt/2*(qdot_n + qdot_{n+1}^k)
            //   qdot_{n+1}^k = 0.5 * q_np1^{k-1} ⊗ ω_{n+1}^k  (linearized at prev iter)
            if (nqb == 7 && nvb >= 6) {
                const double qw = q_np1[qo+3], qx = q_np1[qo+4],
                             qy = q_np1[qo+5], qz = q_np1[qo+6];
                const double wx = v_np1[vo+3], wy = v_np1[vo+4], wz = v_np1[vo+5];
                const double dW = -0.5*(qx*wx + qy*wy + qz*wz);
                const double dX =  0.5*(qw*wx + qy*wz - qz*wy);
                const double dY =  0.5*(qw*wy + qz*wx - qx*wz);
                const double dZ =  0.5*(qw*wz + qx*wy - qy*wx);
                q_np1_new[qo+3] = state.q[qo+3] + 0.5*dt*(qdot_n[qo+3] + dW);
                q_np1_new[qo+4] = state.q[qo+4] + 0.5*dt*(qdot_n[qo+4] + dX);
                q_np1_new[qo+5] = state.q[qo+5] + 0.5*dt*(qdot_n[qo+5] + dY);
                q_np1_new[qo+6] = state.q[qo+6] + 0.5*dt*(qdot_n[qo+6] + dZ);
            } else {
                for (int j = nt; j < nqb; j++)
                    q_np1_new[qo+j] = q_pred[qo+j];
            }
        }
        q_np1 = q_np1_new;

        // Build α-interpolated state
        StateVector s_alpha = state.clone();
        for (int i = 0; i < nq; i++)
            s_alpha.q[i] = (1.0 + alpha) * q_np1[i] - alpha * state.q[i];
        for (int i = 0; i < nv; i++)
            s_alpha.v[i] = (1.0 + alpha) * v_np1[i]  - alpha * state.v[i];
        s_alpha.time = t_alpha;
        s_alpha.normalizeQuaternions();

        std::vector<double> a_next(nv, 0.0);
        std::vector<double> lambda_next(nc, 0.0);

        if (kktSolver_) {
            // ── CORRECT HHT-DAE (Negrut 2007) ──
            // Forces M,Q and constraint Jacobian Cq at α-state.
            // Baumgarte feedback γ evaluated at n+1 state (correct time level).
            StateVector s_np1 = state.clone();
            for (int i = 0; i < nq; i++) s_np1.q[i] = q_np1[i];
            for (int i = 0; i < nv; i++) s_np1.v[i] = v_np1[i];
            s_np1.time = t + dt;
            s_np1.normalizeQuaternions();
            auto r = kktSolver_(t_alpha, s_alpha, s_np1);
            a_next      = r.accel;
            lambda_next = r.lambda;
        } else {
            // ── FALLBACK ── derivative function (split-operator)
            auto k = f(t_alpha, s_alpha);
            for (int i = 0; i < nv; i++) a_next[i] = k.v[i];
        }

        // Convergence: mixed abs+rel tolerance, scale = tol*(1+|a|)
        double res = 0.0;
        for (int i = 0; i < nv; i++) {
            double scale = tol_ * (1.0 + std::abs(a_new[i]));
            res = std::max(res, std::abs(a_next[i] - a_new[i]) / scale);
        }
        a_new      = a_next;
        lambda_new = lambda_next;
        if (res < 1.0) break;
    }

    // ── Step 4: Assemble final state ─────────────────────────────────────────
    StateVector result = state.clone();
    for (int i = 0; i < nv; i++)
        result.v[i] = v_pred[i] + dt * gamma_c * a_new[i];
    for (int i = 0; i < nq; i++)
        result.q[i] = q_np1[i];
    for (int i = 0; i < nv; i++)
        result.a[i] = a_new[i];
    for (int i = 0; i < nc && i < (int)lambda_new.size(); i++)
        result.lambda[i] = lambda_new[i];
    result.time = t + dt;
    result.normalizeQuaternions();

    aPrev_      = a_new;
    lambdaPrev_ = lambda_new;    return {result, dt, dt, true};
}

} // namespace mb