#pragma once
#include "ConstraintSolver.h"

namespace mb {

/**
 * Direct solver using LU decomposition with partial pivoting.
 *
 * Solves the augmented system directly by forming the full matrix
 * and using LU factorization with partial pivoting.
 *
 * Supports optional regularization for redundant constraints:
 *   [M      Cq^T ] [a     ]   [Q    ]
 *   [Cq    −ε·I  ] [lambda] = [gamma]
 *
 * Pre-allocates all work buffers on first call and reuses them.
 *
 * Best for:
 * - Small to medium systems (< 1000 DOF)
 * - High accuracy requirements
 */
class DirectSolver : public ConstraintSolver {
public:
    DirectSolver(const SolverConfig& config = {}, double epsilon = 0.0);

    SolverResult solve(
        const MatrixN& M,
        const MatrixN& Cq,
        const std::vector<double>& Q,
        const std::vector<double>& gamma,
        const std::vector<double>& initialGuess = {}
    ) override;

    /**
     * Solve using Schur complement method.
     * More efficient when m << n (few constraints).
     *
     * Eliminates a:
     *   a = M^{-1}*(Q - Cq^T*lambda)
     *   Cq*M^{-1}*Cq^T*lambda = Cq*M^{-1}*Q - gamma
     */
    SolverResult solveSchurComplement(
        const MatrixN& M,
        const MatrixN& Cq,
        const std::vector<double>& Q,
        const std::vector<double>& gamma
    );

private:
    double epsilon_; // Regularization parameter

    // Pre-allocated buffers
    int cachedSize_ = -1;
    std::vector<double> A_;  // Flat column-major augmented matrix
    std::vector<double> b_;  // RHS vector
    std::vector<int> P_;     // Pivot permutation
    std::vector<double> x_;  // Solution

    void ensureBuffers(int totalSize);

    // Residual helpers
    double computeResidualA(const MatrixN& M, const MatrixN& Cq,
                            const std::vector<double>& Q,
                            const std::vector<double>& a,
                            const std::vector<double>& lambda) const;
    double computeResidualC(const MatrixN& Cq,
                            const std::vector<double>& a,
                            const std::vector<double>& gamma) const;
};

} // namespace mb
