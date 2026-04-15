#pragma once
#include "ConstraintSolver.h"
#include "mb/core/Body.h"
#include "mb/constraints/Constraint.h"
#include "mb/core/State.h"
#include <vector>
#include <functional>

namespace mb {

/**
 * Newton-Raphson solver for nonlinear constraint systems.
 *
 * Can act as:
 * 1. An iterative solver for the augmented DAE system (with warm-start + line search)
 * 2. A nonlinear position-level constraint projector: C(q) = 0
 *
 * Solves: C(q) = 0  using  q_{k+1} = q_k - J^{+} * C(q_k)
 */
class NewtonRaphsonSolver : public ConstraintSolver {
public:
    struct Config {
        int maxIterations = 50;
        double tolerance = 1e-12;
        bool useLineSearch = true;
        double lineSearchAlpha = 1e-4;  // Armijo condition parameter
        double lineSearchBeta = 0.5;    // Backtracking contraction factor
    };

    NewtonRaphsonSolver();
    explicit NewtonRaphsonSolver(const Config& config);
    explicit NewtonRaphsonSolver(const SolverConfig& config);

    /** Enable/disable line search with optional Armijo/backtracking params */
    void setLineSearch(bool enabled, double alpha = -1.0, double beta = -1.0);

    /**
     * Solve the augmented DAE system iteratively with Newton-Raphson.
     * Supports warm-starting from previous solution and optional line search.
     */
    SolverResult solve(
        const MatrixN& M,
        const MatrixN& Cq,
        const std::vector<double>& Q,
        const std::vector<double>& gamma,
        const std::vector<double>& initialGuess = {}
    ) override;

    /**
     * Solve nonlinear position constraint satisfaction: C(q) = 0
     *
     * @param q        Current generalized coordinates
     * @param computeC Computes constraint violation C(q)
     * @param computeJ Computes constraint Jacobian J = dC/dq
     * @return Corrected q, iterations, converged flag
     */
    struct PositionResult {
        std::vector<double> q;
        int iterations;
        bool converged;
    };

    PositionResult solvePositionConstraints(
        const std::vector<double>& q,
        std::function<std::vector<double>(const std::vector<double>&)> computeC,
        std::function<MatrixN(const std::vector<double>&)> computeJ
    );

    /**
     * Convenience overload: project constraints directly on bodies/state.
     */
    bool solvePositionConstraints(
        std::vector<Body*>& bodies,
        const std::vector<Constraint*>& constraints,
        StateVector& state,
        int maxIter = -1,
        double tol = -1.0
    );

private:
    Config nrConfig_;

    /** Solve least-squares: min ||J*x - b||² via normal equations + Tikhonov */
    std::vector<double> solveLeastSquares(const MatrixN& J,
                                          const std::vector<double>& b);

    /** Backtracking line search for linear system residual */
    double lineSearch(const MatrixN& A,
                      const std::vector<double>& b,
                      const std::vector<double>& x,
                      const std::vector<double>& dx);

    /** Line search for constraint satisfaction */
    double lineSearchConstraints(
        const std::vector<double>& q,
        const std::vector<double>& dq,
        std::function<std::vector<double>(const std::vector<double>&)> computeC
    );
};

} // namespace mb
