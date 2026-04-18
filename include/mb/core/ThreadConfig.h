#pragma once
#include <omp.h>
#include <cstdlib>
#include <algorithm>

namespace mb {

/// Global threading configuration for MBC++ library.
/// Set thread count once at startup; all parallel loops use it.
class ThreadConfig {
public:
    /// Set the number of threads for all parallel operations.
    /// Pass 0 to use all available hardware threads.
    static void setNumThreads(int n) {
        if (n <= 0) n = omp_get_max_threads();
        numThreads_ = n;
        omp_set_num_threads(n);
        // Also limit OpenBLAS/LAPACK internal threads
        auto s = std::to_string(n);
        setenv("OPENBLAS_NUM_THREADS", s.c_str(), 1);
        setenv("MKL_NUM_THREADS", s.c_str(), 1);
        setenv("BLAS_NUM_THREADS", s.c_str(), 1);
    }

    /// Get current thread count setting.
    static int numThreads() { return numThreads_; }

    /// Check if parallelism is enabled (more than 1 thread).
    static bool isParallel() { return numThreads_ > 1; }

private:
    static inline int numThreads_ = 1;
};

} // namespace mb
