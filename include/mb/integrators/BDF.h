#pragma once
#include "TimeIntegrator.h"
#include "mb/solvers/ConstraintSolver.h"
#include <functional>

namespace mb {

/**
 * Backward Euler (BDF-1) implicit integrator
 */
class BackwardEuler : public TimeIntegrator {
public:
    BackwardEuler(const IntegratorConfig& config = {}) : TimeIntegrator(config) {}
    StepResult step(double t, StateVector& state, double dt,
                    DerivativeFunction& f) override;
};

/**
 * BDF-2 (2nd order implicit)
 */
class BDF2 : public TimeIntegrator {
public:
    BDF2(const IntegratorConfig& config = {}) : TimeIntegrator(config) {}
    StepResult step(double t, StateVector& state, double dt,
                    DerivativeFunction& f) override;

private:
    bool hasHistory_ = false;
    StateVector prevState_;
};

/**
 * Generalized-α integrator for structural dynamics
 */
class GeneralizedAlpha : public TimeIntegrator {
public:
    GeneralizedAlpha(double rhoInfinity = 0.9, const IntegratorConfig& config = {});
    StepResult step(double t, StateVector& state, double dt,
                    DerivativeFunction& f) override;

private:
    double alphaM_, alphaF_, beta_, gammaGA_;
};

/**
 * HHT-α (Hilber-Hughes-Taylor) integrator for rigid multibody DAE systems.
 *
 * Extension of Newmark-β with numerical dissipation for high-frequency modes.
 * EOM evaluated at interpolated (α-weighted) state to achieve unconditional
 * stability with controllable damping.
 *
 *   α  ∈ [-1/3, 0]  (α=0 → standard Newmark-β, no extra damping)
 *   γ  = 1/2 - α    (unconditionally stable)
 *   β  = (1 - α)²/4 (second-order accurate)
 *
 * Uses fixed-point corrector iteration with optional coupled DAE solve:
 *   - COUPLED mode (setKKTSolver called): M*a + Cq^T*λ = Q, Cq*a = γ
 *     solved simultaneously at the α-interpolated state each iteration.
 *     Constraint forces λ are consistent with the dynamics at every step.
 *   - FALLBACK mode: standard derivative function (split-operator).
 * Position update: trapezoidal rule on qdot (second-order).
 */
class HHTAlpha : public TimeIntegrator {
public:
    /// @param alpha  Dissipation parameter ∈ [-1/3, 0]  (typical: -0.05 .. -0.1)
    /// @param maxIter   Fixed-point corrector iterations
    /// @param tol       Convergence tolerance for accelerations
    explicit HHTAlpha(double alpha = -0.05,
                      int    maxIter = 5,
                      double tol    = 1e-6,
                      const IntegratorConfig& config = {});

    StepResult step(double t, StateVector& state, double dt,
                    DerivativeFunction& f) override;

    void invalidateCache() override { hasHistory_ = false; }

    /// HHT produces a velocity-consistent state internally; skip outer projection.
    bool needsVelocityProjection() const override { return false; }

    // ── Coupled DAE interface ────────────────────────────────────────────────
    /// Correct HHT-DAE (Negrut 2007): forces M,Q evaluated at α-state,
    /// constraints Cq,γ enforced at n+1 state.
    /// Signature: (t_alpha, s_alpha, s_np1) → KKTResult
    using KKTSolverFn = std::function<KKTResult(double t_alpha,
                                                StateVector& s_alpha,
                                                StateVector& s_np1)>;

    /// Attach a coupled KKT solver (provided by MultibodySystem::step).
    /// When set, the corrector loop solves:
    ///   M(q_α)*a + Cq(q_{n+1})^T*λ = Q(t_α, q_α, v_α)
    ///   Cq(q_{n+1})*a               = γ(q_{n+1}, v_{n+1})
    /// This is the standard HHT-DAE formulation: forces at α, constraints at n+1.
    void setKKTSolver(KKTSolverFn fn) { kktSolver_ = std::move(fn); }

    /// Directly set aPrev_ after external state modification (e.g., post-projection).
    /// Call this instead of invalidateCache() to preserve Newmark continuity
    /// while updating the predictor base acceleration.
    void setAPrev(const std::vector<double>& a) {
        aPrev_      = a;
        hasHistory_ = true;  // keep valid — don't throw away λPrev_
    }

private:
    double alpha_;
    int    maxIter_;
    double tol_;
    bool   hasHistory_ = false;
    std::vector<double> aPrev_;       ///< a_n from previous step
    std::vector<double> lambdaPrev_;  ///< λ_n from previous step (coupled mode)
    KKTSolverFn kktSolver_;           ///< Optional coupled KKT solver
};

/**
 * Semi-implicit Euler (symplectic)
 */
class SemiImplicitEuler : public TimeIntegrator {
public:
    SemiImplicitEuler(const IntegratorConfig& config = {}) : TimeIntegrator(config) {}
    StepResult step(double t, StateVector& state, double dt,
                    DerivativeFunction& f) override;
};

/**
 * Velocity Verlet (symplectic, 2nd order)
 */
class VelocityVerlet : public TimeIntegrator {
public:
    VelocityVerlet(const IntegratorConfig& config = {}) : TimeIntegrator(config) {}
    StepResult step(double t, StateVector& state, double dt,
                    DerivativeFunction& f) override;
};

} // namespace mb
