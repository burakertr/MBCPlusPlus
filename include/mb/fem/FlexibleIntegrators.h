#pragma once
#include "mb/fem/FlexibleBody.h"
#include <vector>

namespace mb {

struct FlexStepResult {
    double maxDisplacement;
    double strainEnergy;
    double kineticEnergy;
    double gravitationalPE;
};

// ─── Explicit RK4 Integrator ─────────────────────────────────

class FlexibleBodyIntegrator {
public:
    explicit FlexibleBodyIntegrator(FlexibleBody& body);

    FlexStepResult step(double dt);

private:
    FlexibleBody& body_;
    std::vector<double> MInvDiag_;

    const std::vector<double>& ensureMassInverse();
    std::vector<double> computeAccelerations();
};

// ─── Dormand-Prince 4(5) Adaptive Integrator ─────────────────

class FlexDOPRI45 {
public:
    explicit FlexDOPRI45(FlexibleBody& body);

    double absTol  = 1e-6;
    double relTol  = 1e-4;
    double maxStep = 0.01;
    double minStep = 1e-8;
    double dtCurrent = 1e-4;  ///< current adaptive step-size

    /// Advance by exactly `dtTarget` (may take multiple internal sub-steps)
    FlexStepResult step(double dtTarget);

    int    totalSteps    = 0;
    int    totalRejects  = 0;

private:
    FlexibleBody& body_;
    std::vector<double> MInvDiag_;
    std::vector<double> cachedA_;  ///< FSAL: cached k7 accelerations
    bool hasFSAL_ = false;

    const std::vector<double>& ensureMassInverse();
    std::vector<double> computeAccelerations();
    /// Single DOPRI45 attempt; returns true if accepted
    bool tryStep(double h);
};

// ─── HHT-α Implicit Integrator for ANCF Flexible Bodies ─────
//
// Full HHT-α method (Hilber-Hughes-Taylor) for ANCF flexible bodies.
//
// Equations of motion at time t_{n+1}:
//   M · a_{n+1} + (1+α) · F_int(q_{n+1}, v_{n+1}) - α · F_int(q_n, v_n)
//                = (1+α) · F_ext(t_{n+1}) - α · F_ext(t_n)
//
// Newmark-β update formulas:
//   q_{n+1} = q_n + h·v_n + h²·[(0.5-β)·a_n + β·a_{n+1}]
//   v_{n+1} = v_n + h·[(1-γ)·a_n + γ·a_{n+1}]
//
// Parameters (unconditionally stable for α ∈ [-1/3, 0]):
//   β = (1-α)²/4,  γ = 0.5-α
//
// Newton-Raphson iteration with analytic tangent stiffness:
//   S = M + (1+α)·γ·h·C + (1+α)·β·h²·K
//   S · Δa = -R
//
// where C is the damping matrix and K is the tangent stiffness.

class FlexHHTIntegrator {
public:
    explicit FlexHHTIntegrator(FlexibleBody& body);

    /// HHT-α parameter: α ∈ [-1/3, 0]. Default -0.05 for mild numerical damping.
    double alpha = -0.05;

    /// Newton-Raphson convergence tolerance (on residual norm)
    double newtonTol = 1e-6;

    /// Maximum Newton-Raphson iterations per time step
    int maxNewtonIter = 25;

    /// Use analytic stiffness matrix (true) or finite-difference (false)
    bool useAnalyticStiffness = true;

    /// Finite-difference perturbation for FD stiffness (if useAnalyticStiffness=false)
    double fdEps = 1e-7;

    /// Enable verbose convergence output
    bool verbose = false;

    /// Advance by one time step dt
    FlexStepResult step(double dt);

    /// Get number of Newton iterations from last step
    int lastNewtonIters() const { return lastIters_; }

    /// Get residual norm from last step
    double lastResidualNorm() const { return lastResNorm_; }

    /// Get total step count
    int totalSteps() const { return stepCount_; }

private:
    FlexibleBody& body_;

    // Previous step state
    std::vector<double> aPrev_;      ///< acceleration from previous step
    std::vector<double> Qprev_;      ///< total forces from previous step (for HHT)
    int stepCount_ = 0;
    int lastIters_ = 0;
    double lastResNorm_ = 0;

    /// Map from global DOF to reduced (free) DOF index
    std::vector<int> getFreeDofMap() const;

    /// Dense LU solve: solve A·x = b, returns x. A is modified in-place.
    static std::vector<double> solveDenseLU(std::vector<double>& A,
                                            const std::vector<double>& b, int n);
};

// ─── Legacy Implicit Newmark-β (Newton-Raphson) Integrator ───
// (Kept for backward compatibility)

class ImplicitFlexIntegrator {
public:
    explicit ImplicitFlexIntegrator(FlexibleBody& body);

    double hhtAlpha = 0;
    double newtonTol = 1e-6;
    int maxNewtonIter = 50;
    double fdEps = 1e-5;

    FlexStepResult step(double dt);

    int lastNewtonIters() const { return lastIters_; }
    double lastResidualNorm() const { return lastResNorm_; }
    int totalSteps() const { return stepCount_; }

private:
    FlexibleBody& body_;
    std::vector<double> aPrev_;
    std::vector<double> MInvDiag_;
    int stepCount_ = 0;
    int lastIters_ = 0;
    double lastResNorm_ = 0.0;

    const std::vector<double>& ensureMassInverse();
    std::vector<int> getFreeDofMap() const;

    static std::vector<double> solveDenseLU(std::vector<double>& A,
                                            const std::vector<double>& b, int n);
};

// ─── Static Equilibrium Solver ───────────────────────────────

struct StaticSolveResult {
    bool converged;
    int iterations;
    double finalResidual;
    double maxDisplacement;
    double strainEnergy;
};

struct StaticSolveOptions {
    int maxIter = 50;
    double tol = 1e-6;
    double fdEps = 1e-5;
    bool verbose = false;
    int nLoadSteps = 1;
};

StaticSolveResult solveStaticEquilibrium(FlexibleBody& body,
                                         const StaticSolveOptions& opts = {});

} // namespace mb