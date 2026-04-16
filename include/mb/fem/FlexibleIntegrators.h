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

// ─── Implicit Newmark-β (Newton-Raphson) Integrator ──────────

class ImplicitFlexIntegrator {
public:
    explicit ImplicitFlexIntegrator(FlexibleBody& body);

    double hhtAlpha = 0;
    double newtonTol = 1e-6;
    int maxNewtonIter = 50;
    double fdEps = 1e-5;

    FlexStepResult step(double dt);

private:
    FlexibleBody& body_;
    std::vector<double> aPrev_;
    std::vector<double> MInvDiag_;
    int stepCount_ = 0;

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

// ─── Mesh Generators ─────────────────────────────────────────

/// Generate a box-shaped tetrahedral mesh
GmshMesh generateBoxTetMesh(double Lx, double Ly, double Lz,
                            int nx, int ny, int nz);

/// Generate a cylindrical tetrahedral mesh
GmshMesh generateCylinderTetMesh(double R, double L,
                                  int nR, int nT, int nZ,
                                  double innerRadius = 0);

} // namespace mb
