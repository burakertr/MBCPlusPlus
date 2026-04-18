#include "mb/math/MatrixN.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

// BLAS / LAPACK Fortran interface
extern "C" {
    void dgemm_(const char* transA, const char* transB,
                const int* m, const int* n, const int* k,
                const double* alpha, const double* A, const int* lda,
                const double* B, const int* ldb,
                const double* beta, double* C, const int* ldc);
    void dgemv_(const char* trans, const int* m, const int* n,
                const double* alpha, const double* A, const int* lda,
                const double* x, const int* incx,
                const double* beta, double* y, const int* incy);
    void dgetrf_(const int* m, const int* n, double* A, const int* lda,
                 int* ipiv, int* info);
    void dgetrs_(const char* trans, const int* n, const int* nrhs,
                 const double* A, const int* lda, const int* ipiv,
                 double* B, const int* ldb, int* info);
}

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
    double alpha = 1.0, beta = 0.0;
    int m = rows, n = other.cols, k = cols;
    dgemm_("N", "N", &m, &n, &k, &alpha,
           data.data(), &m, other.data.data(), &other.rows,
           &beta, result.data.data(), &m);
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
    MatrixN LU(n, n);
    LU.data = data;

    std::vector<int> ipiv(n);
    int info = 0;
    dgetrf_(&n, &n, LU.data.data(), &n, ipiv.data(), &info);

    // Extract L and U from packed LU
    MatrixN L = MatrixN::identity(n);
    MatrixN U = MatrixN::zeros(n, n);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j; i++) U.set(i, j, LU.get(i, j));
        for (int i = j + 1; i < n; i++) L.set(i, j, LU.get(i, j));
    }

    // Convert LAPACK 1-based ipiv to 0-based permutation
    std::vector<int> pivot(n);
    for (int i = 0; i < n; i++) pivot[i] = i;
    int sign = 1;
    for (int i = 0; i < n; i++) {
        int swap_idx = ipiv[i] - 1; // LAPACK is 1-based
        if (swap_idx != i) {
            std::swap(pivot[i], pivot[swap_idx]);
            sign = -sign;
        }
    }

    return {L, U, pivot, sign};
}

MatrixN MatrixN::solve(const MatrixN& b) const {
    int n = rows;
    int nrhs = b.cols;

    // Factor
    std::vector<double> A_copy(data);
    std::vector<int> ipiv(n);
    int info = 0;
    dgetrf_(&n, &n, A_copy.data(), &n, ipiv.data(), &info);

    // Solve (dgetrs overwrites B in-place)
    MatrixN X(n, nrhs);
    X.data = b.data;
    char trans = 'N';
    dgetrs_(&trans, &n, &nrhs, A_copy.data(), &n, ipiv.data(),
            X.data.data(), &n, &info);

    return X;
}

double MatrixN::determinant() const {
    auto [L, U, pivot, sign] = luDecompose();
    double det = static_cast<double>(sign);
    for (int i = 0; i < rows; i++) det *= U.get(i, i);
    return det;
}

std::vector<double> MatrixN::solve(const std::vector<double>& b) const {
    int n = rows;
    int one = 1;
    std::vector<double> A_copy(data);
    std::vector<int> ipiv(n);
    int info = 0;
    dgetrf_(&n, &n, A_copy.data(), &n, ipiv.data(), &info);

    std::vector<double> x(b);
    char trans = 'N';
    dgetrs_(&trans, &n, &one, A_copy.data(), &n, ipiv.data(),
            x.data(), &n, &info);
    return x;
}

std::vector<double> MatrixN::multiplyVector(const std::vector<double>& x) const {
    std::vector<double> y(rows, 0.0);
    double alpha = 1.0, beta = 0.0;
    int incx = 1, incy = 1;
    int m = rows, n = cols;
    dgemv_("N", &m, &n, &alpha, data.data(), &m,
           x.data(), &incx, &beta, y.data(), &incy);
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
