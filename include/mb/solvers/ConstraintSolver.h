#pragma once
#include "mb/math/MatrixN.h"
#include <vector>
#include <string>

namespace mb {

/**
 * Result from constraint solver
 */
struct SolverResult {
    std::vector<double> x;  // Solution vector [accelerations; lambda]
    int iterations;
    double residual;
    bool converged;
};

/**
 * Result of a coupled KKT solve (used by HHTAlpha coupled DAE mode).
 * Accelerations are in full-state v-space (size = totalNv).
 * Lambda are the Lagrange multipliers (size = nc).
 */
struct KKTResult {
    std::vector<double> accel;   ///< Full-state v-space accelerations
    std::vector<double> lambda;  ///< Constraint multipliers
};

/**
 * Configuration for constraint solver
 */
struct SolverConfig {
    int maxIterations = 100;
    double tolerance = 1e-10;
    double relaxation = 1.0;    // Relaxation factor (for iterative methods)
    bool warmStart = true;       // Enable warm-starting from previous solution
    bool verbose = false;
};

/**
 * Abstract base class for constraint solvers.
 * Solves the augmented system:
 *   [M   Cq^T] [a]   [Q]
 *   [Cq  0   ] [λ] = [γ]
 */
class ConstraintSolver {
public:
    ConstraintSolver(const SolverConfig& config = {}) : config_(config) {}
    virtual ~ConstraintSolver() = default;

    /**
     * Solve the augmented system for accelerations and Lagrange multipliers.
     */
    virtual SolverResult solve(
        const MatrixN& M,
        const MatrixN& Cq,
        const std::vector<double>& Q,
        const std::vector<double>& gamma,
        const std::vector<double>& initialGuess = {}
    ) = 0;

    /**
     * Solve for accelerations only (given known Lagrange multipliers):
     *   a = M^{-1} * (Q - Cq^T * λ)
     */
    std::vector<double> solveAccelerations(
        const MatrixN& M,
        const MatrixN& Cq,
        const std::vector<double>& Q,
        const std::vector<double>& lambda
    );

    /**
     * Solve for Lagrange multipliers using Schur complement:
     *   S = Cq * M^{-1} * Cq^T
     *   λ = S^{-1} * (Cq * M^{-1} * Q - γ)
     */
    std::vector<double> solveLambda(
        const MatrixN& M,
        const MatrixN& Cq,
        const std::vector<double>& Q,
        const std::vector<double>& gamma
    );

    /** Get the last solution for warm-starting */
    const std::vector<double>& getLastSolution() const { return lastSolution_; }
    bool hasLastSolution() const { return !lastSolution_.empty(); }

    /** Clear cached solution */
    void clearCache() { lastSolution_.clear(); }

    /** Update solver configuration */
    void setConfig(const SolverConfig& config) { config_ = config; }
    const SolverConfig& getConfig() const { return config_; }

protected:
    SolverConfig config_;
    std::vector<double> lastSolution_;
};

} // namespace mb
