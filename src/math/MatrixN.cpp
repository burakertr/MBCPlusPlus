#include "mb/math/MatrixN.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace mb {

// ============== MatrixN ==============

MatrixN::MatrixN(int rows, int cols) : rows(rows), cols(cols), data(rows * cols, 0.0) {}

MatrixN::MatrixN(int rows, int cols, const std::vector<double>& data)
    : rows(rows), cols(cols), data(data) {}

MatrixN MatrixN::zeros(int rows, int cols) {
    return MatrixN(rows, cols);
}

MatrixN MatrixN::identity(int n) {
    MatrixN m(n, n);
    for (int i = 0; i < n; i++) m.set(i, i, 1.0);
    return m;
}

MatrixN MatrixN::columnVector(const double* d, int n) {
    MatrixN m(n, 1);
    for (int i = 0; i < n; i++) m.data[i] = d[i];
    return m;
}

MatrixN MatrixN::columnVector(const std::vector<double>& v) {
    MatrixN m(static_cast<int>(v.size()), 1);
    m.data = v;
    return m;
}

std::vector<double> MatrixN::getColumn(int col) const {
    std::vector<double> v(rows);
    for (int i = 0; i < rows; i++) v[i] = data[col * rows + i];
    return v;
}

void MatrixN::setColumn(int col, const std::vector<double>& v) {
    int n = std::min(rows, static_cast<int>(v.size()));
    for (int i = 0; i < n; i++) data[col * rows + i] = v[i];
}

std::vector<double> MatrixN::getRow(int row) const {
    std::vector<double> v(cols);
    for (int j = 0; j < cols; j++) v[j] = get(row, j);
    return v;
}

MatrixN MatrixN::multiply(const MatrixN& other) const {
    MatrixN result(rows, other.cols);
    for (int j = 0; j < other.cols; j++) {
        for (int k = 0; k < cols; k++) {
            double bkj = other.data[j * other.rows + k];
            if (bkj == 0.0) continue;
            for (int i = 0; i < rows; i++) {
                result.data[j * rows + i] += data[k * rows + i] * bkj;
            }
        }
    }
    return result;
}

MatrixN MatrixN::transpose() const {
    MatrixN result(cols, rows);
    for (int j = 0; j < cols; j++)
        for (int i = 0; i < rows; i++)
            result.data[i * cols + j] = data[j * rows + i];
    return result;
}

MatrixN MatrixN::add(const MatrixN& other) const {
    MatrixN result(rows, cols);
    for (size_t i = 0; i < data.size(); i++)
        result.data[i] = data[i] + other.data[i];
    return result;
}

MatrixN MatrixN::sub(const MatrixN& other) const {
    MatrixN result(rows, cols);
    for (size_t i = 0; i < data.size(); i++)
        result.data[i] = data[i] - other.data[i];
    return result;
}

MatrixN MatrixN::scale(double s) const {
    MatrixN result(rows, cols);
    for (size_t i = 0; i < data.size(); i++)
        result.data[i] = data[i] * s;
    return result;
}

std::tuple<MatrixN, MatrixN, std::vector<int>, int> MatrixN::luDecompose() const {
    int n = rows;
    MatrixN L = MatrixN::identity(n);
    MatrixN U(n, n);
    U.data = data; // copy

    std::vector<int> pivot(n);
    for (int i = 0; i < n; i++) pivot[i] = i;
    int sign = 1;

    for (int k = 0; k < n; k++) {
        // Find pivot
        int maxRow = k;
        double maxVal = std::abs(U.get(k, k));
        for (int i = k + 1; i < n; i++) {
            double v = std::abs(U.get(i, k));
            if (v > maxVal) { maxVal = v; maxRow = i; }
        }

        if (maxRow != k) {
            // Swap rows in U
            for (int j = 0; j < n; j++)
                std::swap(U.data[j * n + k], U.data[j * n + maxRow]);
            // Swap rows in L (only below diagonal)
            for (int j = 0; j < k; j++)
                std::swap(L.data[j * n + k], L.data[j * n + maxRow]);
            std::swap(pivot[k], pivot[maxRow]);
            sign = -sign;
        }

        double ukk = U.get(k, k);
        if (std::abs(ukk) < 1e-14) continue;

        for (int i = k + 1; i < n; i++) {
            double factor = U.get(i, k) / ukk;
            L.set(i, k, factor);
            for (int j = k; j < n; j++) {
                U.set(i, j, U.get(i, j) - factor * U.get(k, j));
            }
        }
    }

    return {L, U, pivot, sign};
}

MatrixN MatrixN::solve(const MatrixN& b) const {
    auto [L, U, pivot, sign] = luDecompose();
    int n = rows;
    int nrhs = b.cols;
    MatrixN X(n, nrhs);

    for (int col = 0; col < nrhs; col++) {
        // Apply pivot to b
        std::vector<double> pb(n);
        for (int i = 0; i < n; i++) pb[i] = b.get(pivot[i], col);

        // Forward substitution: L * y = pb
        std::vector<double> y(n);
        for (int i = 0; i < n; i++) {
            double sum = pb[i];
            for (int j = 0; j < i; j++) sum -= L.get(i, j) * y[j];
            y[i] = sum;
        }

        // Back substitution: U * x = y
        std::vector<double> x(n);
        for (int i = n - 1; i >= 0; i--) {
            double sum = y[i];
            for (int j = i + 1; j < n; j++) sum -= U.get(i, j) * x[j];
            double uii = U.get(i, i);
            x[i] = std::abs(uii) > 1e-14 ? sum / uii : 0.0;
        }

        for (int i = 0; i < n; i++) X.set(i, col, x[i]);
    }

    return X;
}

double MatrixN::determinant() const {
    auto [L, U, pivot, sign] = luDecompose();
    double det = static_cast<double>(sign);
    for (int i = 0; i < rows; i++) det *= U.get(i, i);
    return det;
}

std::vector<double> MatrixN::solve(const std::vector<double>& b) const {
    MatrixN bMat = MatrixN::columnVector(b);
    MatrixN xMat = solve(bMat);
    return xMat.getColumn(0);
}

std::vector<double> MatrixN::multiplyVector(const std::vector<double>& x) const {
    std::vector<double> y(rows, 0.0);
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            y[i] += get(i, j) * x[j];
        }
    }
    return y;
}

MatrixN MatrixN::inverse() const {
    return solve(MatrixN::identity(rows));
}

int MatrixN::rank(double tol) const {
    // Use SVD-like approach via LU
    auto [L, U, pivot, sign] = luDecompose();
    int r = 0;
    int n = std::min(rows, cols);
    for (int i = 0; i < n; i++) {
        if (std::abs(U.get(i, i)) > tol) r++;
    }
    return r;
}

// ============== VectorN ==============

VectorN::VectorN(int n) : size(n), data(n, 0.0) {}

VectorN::VectorN(const std::vector<double>& d) : size(static_cast<int>(d.size())), data(d) {}

VectorN VectorN::zeros(int n) { return VectorN(n); }

double VectorN::dot(const VectorN& other) const {
    double sum = 0;
    for (int i = 0; i < size; i++) sum += data[i] * other.data[i];
    return sum;
}

double VectorN::norm() const { return std::sqrt(dot(*this)); }

VectorN VectorN::add(const VectorN& other) const {
    VectorN result(size);
    for (int i = 0; i < size; i++) result.data[i] = data[i] + other.data[i];
    return result;
}

VectorN VectorN::sub(const VectorN& other) const {
    VectorN result(size);
    for (int i = 0; i < size; i++) result.data[i] = data[i] - other.data[i];
    return result;
}

VectorN VectorN::scale(double s) const {
    VectorN result(size);
    for (int i = 0; i < size; i++) result.data[i] = data[i] * s;
    return result;
}

} // namespace mb
