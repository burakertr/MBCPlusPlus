#include "mb/solvers/NewtonRaphson.h"
#include <cmath>
#include <algorithm>

namespace mb {

// ─── Constructors ───────────────────────────────────────────

NewtonRaphsonSolver::NewtonRaphsonSolver()
    : ConstraintSolver(SolverConfig{50, 1e-12, 1.0, true, false})
    , nrConfig_()
{}

NewtonRaphsonSolver::NewtonRaphsonSolver(const Config& config)
    : ConstraintSolver(SolverConfig{config.maxIterations, config.tolerance, 1.0, true, false})
    , nrConfig_(config)
{}

NewtonRaphsonSolver::NewtonRaphsonSolver(const SolverConfig& config)
    : ConstraintSolver(config)
    , nrConfig_{config.maxIterations, config.tolerance, true, 1e-4, 0.5}
{}

void NewtonRaphsonSolver::setLineSearch(bool enabled, double alpha, double beta) {
    nrConfig_.useLineSearch = enabled;
    if (alpha >= 0) nrConfig_.lineSearchAlpha = alpha;
    if (beta >= 0) nrConfig_.lineSearchBeta = beta;
}

// ─── Augmented DAE solve (iterative Newton) ─────────────────

SolverResult NewtonRaphsonSolver::solve(
    const MatrixN& M,
    const MatrixN& Cq,
    const std::vector<double>& Q,
    const std::vector<double>& gamma,
    const std::vector<double>& initialGuess
) {
    const int n = M.rows;
    const int m = Cq.rows;
    const int totalSize = n + m;

    // Warm-start or initialGuess or zero
    std::vector<double> x(totalSize, 0.0);
    if (!initialGuess.empty() && (int)initialGuess.size() == totalSize) {
        x = initialGuess;
    } else if (config_.warmStart && (int)lastSolution_.size() == totalSize) {
        x = lastSolution_;
    }

    // Build augmented matrix A
    MatrixN A(totalSize, totalSize);
    std::fill(A.data.begin(), A.data.end(), 0.0);

    // Top-left: M
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            A.set(i, j, M.get(i, j));

    // Top-right: Cq^T
    for (int j = 0; j < m; j++)
        for (int i = 0; i < n; i++)
            A.set(i, n + j, Cq.get(j, i));

    // Bottom-left: Cq
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            A.set(n + i, j, Cq.get(i, j));

    // Build RHS b
    std::vector<double> b(totalSize);
    for (int i = 0; i < n; i++) b[i] = Q[i];
    for (int i = 0; i < m; i++) b[n + i] = gamma[i];

    // Newton iteration
    int iterations = 0;
    double residual = std::numeric_limits<double>::infinity();

    for (int iter = 0; iter < nrConfig_.maxIterations; iter++) {
        iterations = iter + 1;

        // Compute residual: r = b - A*x
        std::vector<double> Ax = A.multiplyVector(x);
        std::vector<double> r(totalSize);
        residual = 0;
        for (int i = 0; i < totalSize; i++) {
            r[i] = b[i] - Ax[i];
            residual += r[i] * r[i];
        }
        residual = std::sqrt(residual);

        if (residual < nrConfig_.tolerance) break;

        // Solve A * dx = r
        std::vector<double> dx = A.solve(r);

        // Optional line search
        double alpha = 1.0;
        if (nrConfig_.useLineSearch) {
            alpha = lineSearch(A, b, x, dx);
        }

        // Update: x += alpha * dx
        for (int i = 0; i < totalSize; i++) {
            x[i] += alpha * dx[i];
        }
    }

    lastSolution_ = x;

    return { x, iterations, residual, residual < nrConfig_.tolerance };
}

// ─── Nonlinear position constraint projection (functional) ──

NewtonRaphsonSolver::PositionResult NewtonRaphsonSolver::solvePositionConstraints(
    const std::vector<double>& q,
    std::function<std::vector<double>(const std::vector<double>&)> computeC,
    std::function<MatrixN(const std::vector<double>&)> computeJ
) {
    const int nq = static_cast<int>(q.size());
    std::vector<double> qNew = q;

    int iterations = 0;
    bool converged = false;

    for (int iter = 0; iter < nrConfig_.maxIterations; iter++) {
        iterations = iter + 1;

        // Constraint violation
        std::vector<double> C = computeC(qNew);
        const int mc = static_cast<int>(C.size());

        // Check convergence
        double maxViol = 0;
        for (int i = 0; i < mc; i++) {
            maxViol = std::max(maxViol, std::abs(C[i]));
        }
        if (maxViol < nrConfig_.tolerance) {
            converged = true;
            break;
        }

        // Jacobian
        MatrixN J = computeJ(qNew);

        // dq = -J^+ * C  (least squares)
        std::vector<double> negC(mc);
        for (int i = 0; i < mc; i++) negC[i] = -C[i];
        std::vector<double> dq = solveLeastSquares(J, negC);

        // Line search
        double alpha = 1.0;
        if (nrConfig_.useLineSearch) {
            alpha = lineSearchConstraints(qNew, dq, computeC);
        }

        // Update
        for (int i = 0; i < nq; i++) {
            qNew[i] += alpha * dq[i];
        }
    }

    return { qNew, iterations, converged };
}

// ─── Convenience overload: project on bodies/state ──────────

bool NewtonRaphsonSolver::solvePositionConstraints(
    std::vector<Body*>& bodies,
    const std::vector<Constraint*>& constraints,
    StateVector& state,
    int maxIter,
    double tol
) {
    int mi = (maxIter > 0) ? maxIter : nrConfig_.maxIterations;
    double t = (tol > 0) ? tol : nrConfig_.tolerance;

    for (int iter = 0; iter < mi; iter++) {
        // Compute total violation
        double maxViol = 0;
        int totalEq = 0;
        for (auto* c : constraints) totalEq += c->numEquations();

        std::vector<double> C(totalEq);
        int row = 0;
        for (auto* c : constraints) {
            auto viol = c->computeViolation();
            for (int i = 0; i < c->numEquations(); i++) {
                C[row + i] = viol.position[i];
                maxViol = std::max(maxViol, std::abs(viol.position[i]));
            }
            row += c->numEquations();
        }

        if (maxViol < t) return true;

        // Total velocity DOFs
        int totalNv = state.totalNv;

        // Assemble full Jacobian (same logic as MultibodySystem::assembleJacobian)
        MatrixN J(totalEq, totalNv);
        std::fill(J.data.begin(), J.data.end(), 0.0);

        row = 0;
        for (auto* c : constraints) {
            auto jac = c->computeJacobian();
            auto bodyIds = c->getBodyIds();
            int neq = c->numEquations();

            // Body 1
            if (bodyIds.size() >= 1) {
                int idx = -1;
                for (int b = 0; b < (int)bodies.size(); b++) {
                    if (bodies[b]->id == bodyIds[0]) { idx = b; break; }
                }
                if (idx >= 0) {
                    int off = state.vOffsets[idx];
                    int nvb = state.nvPerBody[idx];
                    for (int i = 0; i < neq; i++)
                        for (int j = 0; j < std::min(nvb, jac.J1.cols); j++)
                            J.set(row + i, off + j, jac.J1.get(i, j));
                }
            }
            // Body 2
            if (bodyIds.size() >= 2) {
                int idx = -1;
                for (int b = 0; b < (int)bodies.size(); b++) {
                    if (bodies[b]->id == bodyIds[1]) { idx = b; break; }
                }
                if (idx >= 0) {
                    int off = state.vOffsets[idx];
                    int nvb = state.nvPerBody[idx];
                    for (int i = 0; i < neq; i++)
                        for (int j = 0; j < std::min(nvb, jac.J2.cols); j++)
                            J.set(row + i, off + j, jac.J2.get(i, j));
                }
            }
            row += neq;
        }

        // Solve dv = -J^+ * C (least squares)
        std::vector<double> negC(totalEq);
        for (int i = 0; i < totalEq; i++) negC[i] = -C[i];
        std::vector<double> dv = solveLeastSquares(J, negC);

        // Apply correction to positions via velocity-like mapping
        for (int b = 0; b < (int)bodies.size(); b++) {
            if (!bodies[b]->isDynamic()) continue;

            int qOff = state.qOffsets[b];
            int vOff = state.vOffsets[b];
            int ndof = state.nvPerBody[b];

            // Translational correction (first 3 velocity DOFs → first 3 position coords)
            for (int i = 0; i < std::min(ndof, 3); i++) {
                state.q[qOff + i] += dv[vOff + i];
            }
            // Quaternion correction from angular velocity delta
            if (ndof >= 6) {
                double wx = dv[vOff + 3];
                double wy = dv[vOff + 4];
                double wz = dv[vOff + 5];
                double q0 = state.q[qOff + 3];
                double q1 = state.q[qOff + 4];
                double q2 = state.q[qOff + 5];
                double q3 = state.q[qOff + 6];
                // dq = 0.5 * E(q) * ω
                state.q[qOff + 3] += 0.5 * (-q1*wx - q2*wy - q3*wz);
                state.q[qOff + 4] += 0.5 * ( q0*wx - q3*wy + q2*wz);
                state.q[qOff + 5] += 0.5 * ( q3*wx + q0*wy - q1*wz);
                state.q[qOff + 6] += 0.5 * (-q2*wx + q1*wy + q0*wz);
                // Re-normalize quaternion
                double nrm = std::sqrt(
                    state.q[qOff+3]*state.q[qOff+3] + state.q[qOff+4]*state.q[qOff+4] +
                    state.q[qOff+5]*state.q[qOff+5] + state.q[qOff+6]*state.q[qOff+6]);
                if (nrm > 1e-15) {
                    double inv = 1.0 / nrm;
                    state.q[qOff+3] *= inv;
                    state.q[qOff+4] *= inv;
                    state.q[qOff+5] *= inv;
                    state.q[qOff+6] *= inv;
                }
            }
        }

        // Sync state back to bodies
        for (int b = 0; b < (int)bodies.size(); b++) {
            state.copyToBody(b, bodies[b]);
        }
    }

    return false;
}

// ─── Least-squares via normal equations + Tikhonov ──────────

std::vector<double> NewtonRaphsonSolver::solveLeastSquares(
    const MatrixN& J,
    const std::vector<double>& b
) {
    MatrixN JT = J.transpose();
    MatrixN JTJ = JT.multiply(J);
    MatrixN JTb = JT.multiply(MatrixN::columnVector(b));

    // Tikhonov regularization for numerical stability
    int n = JTJ.rows;
    for (int i = 0; i < n; i++) {
        JTJ.set(i, i, JTJ.get(i, i) + 1e-10);
    }

    MatrixN xMat = JTJ.solve(JTb);
    std::vector<double> x(n);
    for (int i = 0; i < n; i++) x[i] = xMat.get(i, 0);
    return x;
}

// ─── Backtracking line search (linear system) ───────────────

double NewtonRaphsonSolver::lineSearch(
    const MatrixN& A,
    const std::vector<double>& b,
    const std::vector<double>& x,
    const std::vector<double>& dx
) {
    const int n = static_cast<int>(x.size());
    double alpha = 1.0;

    // Initial residual norm²
    std::vector<double> Ax0 = A.multiplyVector(x);
    double r0Sq = 0;
    for (int i = 0; i < n; i++) {
        double ri = b[i] - Ax0[i];
        r0Sq += ri * ri;
    }

    // Backtracking with Armijo condition
    for (int k = 0; k < 20; k++) {
        std::vector<double> xNew(n);
        for (int i = 0; i < n; i++) xNew[i] = x[i] + alpha * dx[i];

        std::vector<double> AxNew = A.multiplyVector(xNew);
        double rNewSq = 0;
        for (int i = 0; i < n; i++) {
            double ri = b[i] - AxNew[i];
            rNewSq += ri * ri;
        }

        if (rNewSq < r0Sq * (1.0 - nrConfig_.lineSearchAlpha * alpha)) {
            break;
        }
        alpha *= nrConfig_.lineSearchBeta;
    }

    return alpha;
}

// ─── Line search for constraint satisfaction ────────────────

double NewtonRaphsonSolver::lineSearchConstraints(
    const std::vector<double>& q,
    const std::vector<double>& dq,
    std::function<std::vector<double>(const std::vector<double>&)> computeC
) {
    const int n = static_cast<int>(q.size());
    double alpha = 1.0;

    std::vector<double> C0 = computeC(q);
    double c0Sq = 0;
    for (double c : C0) c0Sq += c * c;

    for (int k = 0; k < 20; k++) {
        std::vector<double> qNew(n);
        for (int i = 0; i < n; i++) qNew[i] = q[i] + alpha * dq[i];

        std::vector<double> CNew = computeC(qNew);
        double cNewSq = 0;
        for (double c : CNew) cNewSq += c * c;

        if (cNewSq < c0Sq * (1.0 - nrConfig_.lineSearchAlpha * alpha)) {
            break;
        }
        alpha *= nrConfig_.lineSearchBeta;
    }

    return alpha;
}

} // namespace mb
