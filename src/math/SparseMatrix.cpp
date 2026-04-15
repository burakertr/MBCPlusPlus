#include "mb/math/SparseMatrix.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace mb {

SparseMatrix::SparseMatrix(int rows, int cols) : rows_(rows), cols_(cols) {}

void SparseMatrix::set(int row, int col, double value) {
    // Check for existing entry
    for (auto& e : entries_) {
        if (e.row == row && e.col == col) {
            e.value = value;
            cscDirty_ = true;
            return;
        }
    }
    entries_.push_back({row, col, value});
    cscDirty_ = true;
}

double SparseMatrix::get(int row, int col) const {
    for (const auto& e : entries_) {
        if (e.row == row && e.col == col) return e.value;
    }
    return 0.0;
}

void SparseMatrix::addValue(int row, int col, double value) {
    for (auto& e : entries_) {
        if (e.row == row && e.col == col) {
            e.value += value;
            cscDirty_ = true;
            return;
        }
    }
    entries_.push_back({row, col, value});
    cscDirty_ = true;
}

void SparseMatrix::clear() {
    entries_.clear();
    cscDirty_ = true;
}

MatrixN SparseMatrix::toDense() const {
    MatrixN M(rows_, cols_);
    for (const auto& e : entries_) {
        M.set(e.row, e.col, e.value);
    }
    return M;
}

void SparseMatrix::buildCSC() const {
    if (!cscDirty_) return;

    // Sort entries by column, then row
    std::vector<Entry> sorted = entries_;
    std::sort(sorted.begin(), sorted.end(), [](const Entry& a, const Entry& b) {
        return a.col < b.col || (a.col == b.col && a.row < b.row);
    });

    cscColPtr_.assign(cols_ + 1, 0);
    cscRowIdx_.clear();
    cscValues_.clear();

    for (const auto& e : sorted) {
        if (std::abs(e.value) < 1e-20) continue;
        cscColPtr_[e.col + 1]++;
        cscRowIdx_.push_back(e.row);
        cscValues_.push_back(e.value);
    }

    // Cumulative sum
    for (int j = 0; j < cols_; j++) {
        cscColPtr_[j + 1] += cscColPtr_[j];
    }

    cscDirty_ = false;
}

std::vector<double> SparseMatrix::multiplyVector(const std::vector<double>& x) const {
    buildCSC();
    return multiplyCSC(x);
}

std::vector<double> SparseMatrix::multiplyCSC(const std::vector<double>& x) const {
    std::vector<double> y(rows_, 0.0);
    for (int j = 0; j < cols_; j++) {
        double xj = x[j];
        if (xj == 0.0) continue;
        for (int k = cscColPtr_[j]; k < cscColPtr_[j + 1]; k++) {
            y[cscRowIdx_[k]] += cscValues_[k] * xj;
        }
    }
    return y;
}

std::vector<double> SparseMatrix::solveCG(
    const std::vector<double>& b, int maxIter, double tol
) const {
    int n = rows_;
    std::vector<double> x(n, 0.0);
    std::vector<double> r = b; // r = b - A*x, but x=0 so r=b
    std::vector<double> p = r;
    double rsOld = 0;
    for (int i = 0; i < n; i++) rsOld += r[i] * r[i];

    for (int iter = 0; iter < maxIter; iter++) {
        auto Ap = multiplyVector(p);
        double pAp = 0;
        for (int i = 0; i < n; i++) pAp += p[i] * Ap[i];
        if (std::abs(pAp) < 1e-20) break;

        double alpha = rsOld / pAp;
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        double rsNew = 0;
        for (int i = 0; i < n; i++) rsNew += r[i] * r[i];
        if (std::sqrt(rsNew) < tol) break;

        double beta = rsNew / rsOld;
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }
        rsOld = rsNew;
    }
    return x;
}

std::vector<double> SparseMatrix::solveBiCGSTAB(
    const std::vector<double>& b, int maxIter, double tol
) const {
    int n = rows_;
    std::vector<double> x(n, 0.0);
    std::vector<double> r = b;
    std::vector<double> r0hat = r;

    double rho = 1, alpha = 1, omega = 1;
    std::vector<double> v(n, 0.0), p(n, 0.0), s(n), t(n);

    for (int iter = 0; iter < maxIter; iter++) {
        double rhoNew = 0;
        for (int i = 0; i < n; i++) rhoNew += r0hat[i] * r[i];
        if (std::abs(rhoNew) < 1e-20) break;

        double beta = (rhoNew / rho) * (alpha / omega);
        rho = rhoNew;

        for (int i = 0; i < n; i++)
            p[i] = r[i] + beta * (p[i] - omega * v[i]);

        v = multiplyVector(p);
        double r0v = 0;
        for (int i = 0; i < n; i++) r0v += r0hat[i] * v[i];
        if (std::abs(r0v) < 1e-20) break;

        alpha = rho / r0v;
        for (int i = 0; i < n; i++)
            s[i] = r[i] - alpha * v[i];

        // Check convergence
        double sNorm = 0;
        for (int i = 0; i < n; i++) sNorm += s[i] * s[i];
        if (std::sqrt(sNorm) < tol) {
            for (int i = 0; i < n; i++) x[i] += alpha * p[i];
            break;
        }

        t = multiplyVector(s);
        double tt = 0, ts = 0;
        for (int i = 0; i < n; i++) { tt += t[i] * t[i]; ts += t[i] * s[i]; }
        omega = (std::abs(tt) > 1e-20) ? ts / tt : 0;

        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i] + omega * s[i];
            r[i] = s[i] - omega * t[i];
        }

        double rNorm = 0;
        for (int i = 0; i < n; i++) rNorm += r[i] * r[i];
        if (std::sqrt(rNorm) < tol) break;
    }
    return x;
}

} // namespace mb
