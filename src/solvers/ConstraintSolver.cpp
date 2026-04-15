#include "mb/solvers/ConstraintSolver.h"
#include <cmath>

namespace mb {

std::vector<double> ConstraintSolver::solveAccelerations(
    const MatrixN& M,
    const MatrixN& Cq,
    const std::vector<double>& Q,
    const std::vector<double>& lambda
) {
    int n = M.rows;
    int m = static_cast<int>(lambda.size());

    // a = M^{-1} * (Q - Cq^T * lambda)
    MatrixN CqT = Cq.transpose();
    MatrixN lambdaVec = MatrixN::columnVector(lambda);
    MatrixN constraintForces = CqT.multiply(lambdaVec);

    std::vector<double> rhs(n);
    for (int i = 0; i < n; i++) {
        rhs[i] = Q[i] - constraintForces.get(i, 0);
    }

    return M.solve(rhs);
}

std::vector<double> ConstraintSolver::solveLambda(
    const MatrixN& M,
    const MatrixN& Cq,
    const std::vector<double>& Q,
    const std::vector<double>& gamma
) {
    int m = Cq.rows;

    // Schur complement: S = Cq * M^{-1} * Cq^T
    MatrixN Minv = M.inverse();
    MatrixN CqMinv = Cq.multiply(Minv);
    MatrixN CqT = Cq.transpose();
    MatrixN S = CqMinv.multiply(CqT);

    // rhs = gamma - Cq * M^{-1} * Q
    MatrixN MinvQ = Minv.multiply(MatrixN::columnVector(Q));
    MatrixN CqMinvQ = Cq.multiply(MinvQ);

    std::vector<double> rhs(m);
    for (int i = 0; i < m; i++) {
        rhs[i] = gamma[i] - CqMinvQ.get(i, 0);
    }

    return S.solve(rhs);
}

} // namespace mb
