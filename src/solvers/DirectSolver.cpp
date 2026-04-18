#include "mb/solvers/DirectSolver.h"
#include <cmath>
#include <algorithm>

// LAPACK Fortran interface
extern "C" {
    void dgetrf_(const int* m, const int* n, double* A, const int* lda,
                 int* ipiv, int* info);
    void dgetrs_(const char* trans, const int* n, const int* nrhs,
                 const double* A, const int* lda, const int* ipiv,
                 double* B, const int* ldb, int* info);
}

namespace mb {

DirectSolver::DirectSolver(const SolverConfig& config, double epsilon)
    : ConstraintSolver(config), epsilon_(epsilon) {}

void DirectSolver::ensureBuffers(int totalSize) {
    if (totalSize == cachedSize_) return;
    cachedSize_ = totalSize;
    A_.resize(totalSize * totalSize);
    b_.resize(totalSize);
    P_.resize(totalSize);
    x_.resize(totalSize);
}

SolverResult DirectSolver::solve(
    const MatrixN& M,
    const MatrixN& Cq,
    const std::vector<double>& Q,
    const std::vector<double>& gamma,
    const std::vector<double>& /*initialGuess*/
) {
    const int n = M.rows;    // velocity DOFs
    const int m = Cq.rows;   // constraints
    const int totalSize = n + m;

    ensureBuffers(totalSize);

    // ── Zero out flat column-major augmented matrix ──
    std::fill(A_.begin(), A_.end(), 0.0);

    // Top-left: M (n×n)
    for (int j = 0; j < n; j++) {
        const int dstCol = j * totalSize;
        const int srcCol = j * n;
        for (int i = 0; i < n; i++) {
            A_[dstCol + i] = M.data[srcCol + i];
        }
    }

    // Top-right: Cq^T → A[i, n+j] = Cq(j, i)
    // Cq is m×n column-major: Cq(r,c) = Cq.data[c * m + r]
    for (int j = 0; j < m; j++) {
        const int dstCol = (n + j) * totalSize;
        for (int i = 0; i < n; i++) {
            A_[dstCol + i] = Cq.data[i * m + j];
        }
    }

    // Bottom-left: Cq → A[n+r, c] = Cq(r, c)
    for (int c = 0; c < n; c++) {
        const int dstCol = c * totalSize;
        const int srcCol = c * m;
        for (int r = 0; r < m; r++) {
            A_[dstCol + n + r] = Cq.data[srcCol + r];
        }
    }

    // Regularization: -ε·I in bottom-right block
    if (epsilon_ > 0) {
        for (int i = 0; i < m; i++) {
            A_[(n + i) * totalSize + (n + i)] = -epsilon_;
        }
    }

    // ── RHS ──
    for (int i = 0; i < n; i++) b_[i] = Q[i];
    for (int i = 0; i < m; i++) b_[n + i] = gamma[i];

    // ── LAPACK LU factorization + solve ──
    int info = 0;
    int one = 1;
    dgetrf_(&totalSize, &totalSize, A_.data(), &totalSize, P_.data(), &info);

    // Copy RHS into x_ (dgetrs overwrites RHS in-place)
    std::copy(b_.begin(), b_.begin() + totalSize, x_.begin());

    char trans = 'N';
    dgetrs_(&trans, &totalSize, &one, A_.data(), &totalSize, P_.data(),
            x_.data(), &totalSize, &info);

    // Store for warm-starting
    lastSolution_.assign(x_.begin(), x_.end());

    return { std::vector<double>(x_.begin(), x_.end()), 1, 0.0, true };
}

SolverResult DirectSolver::solveSchurComplement(
    const MatrixN& M,
    const MatrixN& Cq,
    const std::vector<double>& Q,
    const std::vector<double>& gamma
) {
    const int n = M.rows;
    const int m = Cq.rows;

    if (m == 0) {
        // No constraints → just solve M*a = Q
        auto a = M.solve(Q);
        lastSolution_ = a;
        return { a, 1, 0.0, true };
    }

    // M^{-1}
    MatrixN Minv = M.inverse();

    // S = Cq * M^{-1} * Cq^T
    MatrixN CqMinv = Cq.multiply(Minv);
    MatrixN CqT = Cq.transpose();
    MatrixN S = CqMinv.multiply(CqT);

    // rhs = Cq * M^{-1} * Q - gamma
    std::vector<double> MinvQ = Minv.multiplyVector(Q);
    std::vector<double> CqMinvQ_vec = Cq.multiplyVector(MinvQ);

    std::vector<double> rhs(m);
    for (int i = 0; i < m; i++) {
        rhs[i] = CqMinvQ_vec[i] - gamma[i];
    }

    // Solve for lambda
    std::vector<double> lambda = S.solve(rhs);

    // Back-substitute: a = M^{-1} * (Q - Cq^T * lambda)
    std::vector<double> CqTLambda = CqT.multiplyVector(lambda);
    std::vector<double> rhsA(n);
    for (int i = 0; i < n; i++) {
        rhsA[i] = Q[i] - CqTLambda[i];
    }
    std::vector<double> a = Minv.multiplyVector(rhsA);

    // Combine solution
    std::vector<double> x(n + m);
    std::copy(a.begin(), a.end(), x.begin());
    std::copy(lambda.begin(), lambda.end(), x.begin() + n);

    // Compute residual
    double residualA = computeResidualA(M, Cq, Q, a, lambda);
    double residualC = computeResidualC(Cq, a, gamma);
    double residual = std::sqrt(residualA + residualC);

    lastSolution_ = x;

    return { x, 1, residual, residual < config_.tolerance };
}

double DirectSolver::computeResidualA(
    const MatrixN& M, const MatrixN& Cq,
    const std::vector<double>& Q,
    const std::vector<double>& a,
    const std::vector<double>& lambda
) const {
    int n = M.rows;
    std::vector<double> Ma = M.multiplyVector(a);
    MatrixN CqT = Cq.transpose();
    std::vector<double> CqTL = CqT.multiplyVector(lambda);

    double sum = 0;
    for (int i = 0; i < n; i++) {
        double diff = Ma[i] + CqTL[i] - Q[i];
        sum += diff * diff;
    }
    return sum;
}

double DirectSolver::computeResidualC(
    const MatrixN& Cq,
    const std::vector<double>& a,
    const std::vector<double>& gamma
) const {
    int m = Cq.rows;
    std::vector<double> CqA = Cq.multiplyVector(a);

    double sum = 0;
    for (int i = 0; i < m; i++) {
        double diff = CqA[i] - gamma[i];
        sum += diff * diff;
    }
    return sum;
}

} // namespace mb
