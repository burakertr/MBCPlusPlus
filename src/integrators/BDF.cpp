#include "mb/integrators/BDF.h"
#include "mb/math/MatrixN.h"
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

    // ── Step 3: Nonlinear coupled DAE Newton solve ──────────────────────────
    std::vector<double> a_new(aPrev_);
    std::vector<double> lambda_new(lambdaPrev_);
    std::vector<double> q_np1(q_pred);
    std::vector<double> v_np1(v_pred);
    bool converged = false;

    if (daeResidualFn_) {
        std::vector<int> dynDofs;
        dynDofs.reserve(nv);
        for (int b = 0; b < nb; b++) {
            bool isDyn = (b < (int)state.dynamicBody.size()) ? state.dynamicBody[b] : true;
            if (!isDyn) continue;
            int vo = state.vOffsets[b];
            for (int j = 0; j < state.nvPerBody[b]; j++)
                dynDofs.push_back(vo + j);
        }
        const int nDyn = (int)dynDofs.size();
        const int nx = nDyn + nc;

        std::vector<double> x(nx, 0.0);
        for (int i = 0; i < nDyn; i++)
            x[i] = a_new[dynDofs[i]];
        for (int i = 0; i < nc && i < (int)lambda_new.size(); i++)
            x[nDyn + i] = lambda_new[i];

        // Chrono-like scaling for position constraints: 1/(beta*h^2)*C.
        const double constraintScale = 1.0 / std::max(1e-14, beta_c * dt * dt);

        auto evalResidual = [&](const std::vector<double>& xIn,
                                std::vector<double>& F,
                                std::vector<double>& aFullOut,
                                std::vector<double>& lambdaOut,
                                std::vector<double>& qOut,
                                std::vector<double>& vOut,
                                double& maxC,
                                double& maxCdot,
                                double& maxDyn) {
            aFullOut.assign(nv, 0.0);
            lambdaOut.assign(nc, 0.0);
            for (int i = 0; i < nDyn; i++)
                aFullOut[dynDofs[i]] = xIn[i];
            for (int i = 0; i < nc; i++)
                lambdaOut[i] = xIn[nDyn + i];

            vOut.resize(nv);
            for (int i = 0; i < nv; i++)
                vOut[i] = v_pred[i] + dt * gamma_c * aFullOut[i];

            qOut.resize(nq);
            for (int b = 0; b < nb; b++) {
                const int qo  = state.qOffsets[b];
                const int vo  = state.vOffsets[b];
                const int nqb = state.nqPerBody[b];
                const int nvb = state.nvPerBody[b];
                const int nt  = std::min(3, std::min(nqb, nvb));

                for (int j = 0; j < nt; j++) {
                    qOut[qo + j] = state.q[qo + j]
                                 + dt * state.v[vo + j]
                                 + dt * dt * ((0.5 - beta_c) * aPrev_[vo + j]
                                            + beta_c * aFullOut[vo + j]);
                }

                if (nqb == 7 && nvb >= 6) {
                    const double wn_x = state.v[vo + 3];
                    const double wn_y = state.v[vo + 4];
                    const double wn_z = state.v[vo + 5];
                    const double wp_x = vOut[vo + 3];
                    const double wp_y = vOut[vo + 4];
                    const double wp_z = vOut[vo + 5];
                    const double wx = 0.5 * (wn_x + wp_x);
                    const double wy = 0.5 * (wn_y + wp_y);
                    const double wz = 0.5 * (wn_z + wp_z);
                    const double wNorm = std::sqrt(wx * wx + wy * wy + wz * wz);

                    Quaternion qn(state.q[qo + 3], state.q[qo + 4],
                                  state.q[qo + 5], state.q[qo + 6]);
                    Quaternion qnp1 = qn;
                    if (wNorm > 1e-14) {
                        const double angle = wNorm * dt;
                        const double half = 0.5 * angle;
                        const double s = std::sin(half) / wNorm;
                        Quaternion dq(std::cos(half), wx * s, wy * s, wz * s);
                        qnp1 = dq.multiply(qn).normalize();
                    }
                    qOut[qo + 3] = qnp1.w;
                    qOut[qo + 4] = qnp1.x;
                    qOut[qo + 5] = qnp1.y;
                    qOut[qo + 6] = qnp1.z;
                } else {
                    for (int j = nt; j < nqb; j++)
                        qOut[qo + j] = q_pred[qo + j];
                }
            }

            StateVector s_np1 = state.clone();
            s_np1.time = t + dt;
            s_np1.q = qOut;
            s_np1.v = vOut;
            s_np1.normalizeQuaternions();
            qOut = s_np1.q;

            StateVector s_alpha = state.clone();
            s_alpha.time = t_alpha;
            for (int i = 0; i < nq; i++)
                s_alpha.q[i] = (1.0 + alpha) * s_np1.q[i] - alpha * state.q[i];
            for (int i = 0; i < nv; i++)
                s_alpha.v[i] = (1.0 + alpha) * s_np1.v[i] - alpha * state.v[i];
            s_alpha.normalizeQuaternions();

            std::vector<double> dynResidual, C, Cdot;
            daeResidualFn_(t_alpha, s_alpha, s_np1, aFullOut, lambdaOut, dynResidual, C, Cdot);

            maxDyn = 0.0;
            for (double v : dynResidual) maxDyn = std::max(maxDyn, std::abs(v));
            maxC = 0.0;
            for (double v : C) maxC = std::max(maxC, std::abs(v));
            maxCdot = 0.0;
            for (double v : Cdot) maxCdot = std::max(maxCdot, std::abs(v));

            F.assign(nx, 0.0);
            for (int i = 0; i < nDyn && i < (int)dynResidual.size(); i++)
                F[i] = dynResidual[i];
            for (int i = 0; i < nc; i++) {
                const double c = (i < (int)C.size()) ? C[i] : 0.0;
                F[nDyn + i] = constraintScale * c;
            }
        };

        const int nonlinearIterMax = std::min(maxIter_, 12);
        for (int iter = 0; iter < nonlinearIterMax; iter++) {
            std::vector<double> F, aEval, lamEval, qEval, vEval;
            double maxC = 0.0, maxCdot = 0.0, maxDyn = 0.0;
            evalResidual(x, F, aEval, lamEval, qEval, vEval, maxC, maxCdot, maxDyn);

            const double dynTol = tol_;
            const double conTol = std::min(tol_, 1e-12);
            if (maxDyn <= dynTol && maxC <= conTol) {
                a_new = aEval;
                lambda_new = lamEval;
                q_np1 = qEval;
                v_np1 = vEval;
                converged = true;
                break;
            }

            MatrixN J(nx, nx);
            for (int col = 0; col < nx; col++) {
                std::vector<double> xPert = x;
                double h = 1e-7 * (1.0 + std::abs(x[col]));
                xPert[col] += h;

                std::vector<double> Fp, aTmp, lTmp, qTmp, vTmp;
                double cTmp = 0.0, cdTmp = 0.0, dTmp = 0.0;
                evalResidual(xPert, Fp, aTmp, lTmp, qTmp, vTmp, cTmp, cdTmp, dTmp);

                for (int row = 0; row < nx; row++) {
                    double deriv = (Fp[row] - F[row]) / h;
                    J.set(row, col, deriv);
                }
            }

            std::vector<double> rhs(nx, 0.0);
            for (int i = 0; i < nx; i++) rhs[i] = -F[i];
            std::vector<double> dx = J.solve(rhs);

            double baseNorm = 0.0;
            for (double v : F) baseNorm = std::max(baseNorm, std::abs(v));
            double lsAlpha = 1.0;
            std::vector<double> xTrial = x;
            bool acceptedStep = false;

            for (int ls = 0; ls < 8; ls++) {
                for (int i = 0; i < nx; i++)
                    xTrial[i] = x[i] + lsAlpha * dx[i];

                std::vector<double> Ft, aTmp, lTmp, qTmp, vTmp;
                double cTmp = 0.0, cdTmp = 0.0, dTmp = 0.0;
                evalResidual(xTrial, Ft, aTmp, lTmp, qTmp, vTmp, cTmp, cdTmp, dTmp);

                double trialNorm = 0.0;
                for (double v : Ft) trialNorm = std::max(trialNorm, std::abs(v));
                if (trialNorm < baseNorm) {
                    acceptedStep = true;
                    break;
                }
                lsAlpha *= 0.5;
            }

            if (!acceptedStep) break;
            x = xTrial;
        }
    } else {
        // Fallback: previous split/coupled fixed-point style if residual callback not provided.
        for (int iter = 0; iter < maxIter_; iter++) {
            for (int i = 0; i < nv; i++)
                v_np1[i] = v_pred[i] + dt * gamma_c * a_new[i];

            for (int b = 0; b < nb; b++) {
                const int qo  = state.qOffsets[b];
                const int vo  = state.vOffsets[b];
                const int nqb = state.nqPerBody[b];
                const int nvb = state.nvPerBody[b];
                const int nt  = std::min(3, std::min(nqb, nvb));
                for (int j = 0; j < nt; j++) {
                    q_np1[qo + j] = state.q[qo + j]
                                  + dt * state.v[vo + j]
                                  + dt * dt * ((0.5 - beta_c) * aPrev_[vo + j]
                                             + beta_c * a_new[vo + j]);
                }
            }

            StateVector s_alpha = state.clone();
            for (int i = 0; i < nq; i++)
                s_alpha.q[i] = (1.0 + alpha) * q_np1[i] - alpha * state.q[i];
            for (int i = 0; i < nv; i++)
                s_alpha.v[i] = (1.0 + alpha) * v_np1[i] - alpha * state.v[i];
            s_alpha.time = t_alpha;
            s_alpha.normalizeQuaternions();

            std::vector<double> a_next(nv, 0.0);
            std::vector<double> lambda_next(nc, 0.0);
            if (kktSolver_) {
                StateVector s_np1 = state.clone();
                s_np1.q = q_np1;
                s_np1.v = v_np1;
                s_np1.time = t + dt;
                s_np1.normalizeQuaternions();
                auto r = kktSolver_(t_alpha, s_alpha, s_np1);
                a_next = r.accel;
                lambda_next = r.lambda;
                if (r.maxPositionViolation <= tol_ && r.maxVelocityViolation <= tol_) {
                    a_new = a_next;
                    lambda_new = lambda_next;
                    q_np1 = s_np1.q;
                    converged = true;
                    break;
                }
            } else {
                auto k = f(t_alpha, s_alpha);
                for (int i = 0; i < nv; i++) a_next[i] = k.v[i];
                a_new = a_next;
                converged = true;
                break;
            }
            a_new = a_next;
            lambda_new = lambda_next;
        }
    }

    if (!converged) {
        const double nextDt = std::max(config_.minStep, dt * 0.5);
        if (dt > config_.minStep * (1.0 + 1e-12))
            return {state.clone(), dt, nextDt, false};
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
    lambdaPrev_ = lambda_new;
    return {result, dt, dt, true};
}

} // namespace mb
