#pragma once
#include <vector>
#include <cstddef>
#include <string>
#include <tuple>

namespace mb {

/**
 * Dense NxM Matrix (column-major storage)
 * data[col * rows + row]
 */
class MatrixN {
public:
    int rows, cols;
    std::vector<double> data;

    MatrixN() : rows(0), cols(0) {}
    MatrixN(int rows, int cols);
    MatrixN(int rows, int cols, const std::vector<double>& data);

    // Factory methods
    static MatrixN zeros(int rows, int cols);
    static MatrixN identity(int n);
    static MatrixN columnVector(const double* data, int n);
    static MatrixN columnVector(const std::vector<double>& v);

    // Element access
    double get(int row, int col) const { return data[col * rows + row]; }
    void set(int row, int col, double v) { data[col * rows + row] = v; }

    // Column/row access
    std::vector<double> getColumn(int col) const;
    void setColumn(int col, const std::vector<double>& v);
    std::vector<double> getRow(int row) const;

    // Operations
    MatrixN multiply(const MatrixN& other) const;
    MatrixN transpose() const;
    MatrixN add(const MatrixN& other) const;
    MatrixN sub(const MatrixN& other) const;
    MatrixN scale(double s) const;

    // LU decomposition with partial pivoting
    // Returns: (L, U, pivot, sign)
    std::tuple<MatrixN, MatrixN, std::vector<int>, int> luDecompose() const;

    // Solve Ax = b (returns x as column matrix)
    MatrixN solve(const MatrixN& b) const;

    // Solve Ax = b where b is a vector, returns x as vector
    std::vector<double> solve(const std::vector<double>& b) const;

    // Matrix-vector multiply: y = A * x
    std::vector<double> multiplyVector(const std::vector<double>& x) const;

    double determinant() const;
    MatrixN inverse() const;
    int rank(double tol = 1e-10) const;

    // Operators
    MatrixN operator*(const MatrixN& o) const { return multiply(o); }
    MatrixN operator+(const MatrixN& o) const { return add(o); }
    MatrixN operator-(const MatrixN& o) const { return sub(o); }
    MatrixN operator*(double s) const { return scale(s); }
};

/**
 * Simple vector wrapper for compatibility
 */
class VectorN {
public:
    int size;
    std::vector<double> data;

    VectorN() : size(0) {}
    VectorN(int n);
    VectorN(const std::vector<double>& data);

    static VectorN zeros(int n);

    double get(int i) const { return data[i]; }
    void set(int i, double v) { data[i] = v; }

    double dot(const VectorN& other) const;
    double norm() const;
    VectorN add(const VectorN& other) const;
    VectorN sub(const VectorN& other) const;
    VectorN scale(double s) const;

    VectorN operator+(const VectorN& o) const { return add(o); }
    VectorN operator-(const VectorN& o) const { return sub(o); }
    VectorN operator*(double s) const { return scale(s); }
};

} // namespace mb
