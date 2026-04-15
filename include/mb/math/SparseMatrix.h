#pragma once
#include <vector>
#include <cstddef>
#include "MatrixN.h"

namespace mb {

/**
 * Sparse Matrix in COO (Coordinate) format with CSC conversion cache.
 * Supports CG solver (SPD) and BiCGSTAB solver (non-symmetric).
 */
class SparseMatrix {
public:
    int rows_, cols_;

    SparseMatrix(int rows, int cols);

    int getRows() const { return rows_; }
    int getCols() const { return cols_; }

    // Set value at (row, col) - COO format
    void set(int row, int col, double value);
    double get(int row, int col) const;
    
    // Add value at (row, col)
    void addValue(int row, int col, double value);

    // Clear all entries
    void clear();

    // Convert to dense
    MatrixN toDense() const;

    // Sparse matrix-vector multiply: y = A * x
    std::vector<double> multiplyVector(const std::vector<double>& x) const;

    /**
     * Conjugate Gradient solver (for SPD matrices)
     * Solves Ax = b, returns x
     */
    std::vector<double> solveCG(
        const std::vector<double>& b,
        int maxIter = 1000,
        double tol = 1e-10
    ) const;

    /**
     * BiCGSTAB solver (for non-symmetric matrices)
     * Solves Ax = b, returns x
     */
    std::vector<double> solveBiCGSTAB(
        const std::vector<double>& b,
        int maxIter = 1000,
        double tol = 1e-10
    ) const;

private:
    // COO format
    struct Entry {
        int row, col;
        double value;
    };
    std::vector<Entry> entries_;

    // CSC cache
    mutable bool cscDirty_ = true;
    mutable std::vector<int> cscColPtr_;
    mutable std::vector<int> cscRowIdx_;
    mutable std::vector<double> cscValues_;

    void buildCSC() const;
    std::vector<double> multiplyCSC(const std::vector<double>& x) const;
};

} // namespace mb
