#pragma once
#include "mb/fem/ANCFTypes.h"
#include <array>
#include <cmath>
#include <memory>

namespace mb {

// ─── 3×3 matrix helpers (row-major flat array) ───────────────────

namespace mat3util {

inline double det(const double* A) {
    return A[0]*(A[4]*A[8]-A[5]*A[7])
         - A[1]*(A[3]*A[8]-A[5]*A[6])
         + A[2]*(A[3]*A[7]-A[4]*A[6]);
}

inline void inv(const double* A, double* out) {
    double d = det(A);
    if (std::abs(d) < 1e-30) d = 1e-30;
    double invD = 1.0 / d;
    out[0] =  (A[4]*A[8]-A[5]*A[7]) * invD;
    out[1] = -(A[1]*A[8]-A[2]*A[7]) * invD;
    out[2] =  (A[1]*A[5]-A[2]*A[4]) * invD;
    out[3] = -(A[3]*A[8]-A[5]*A[6]) * invD;
    out[4] =  (A[0]*A[8]-A[2]*A[6]) * invD;
    out[5] = -(A[0]*A[5]-A[2]*A[3]) * invD;
    out[6] =  (A[3]*A[7]-A[4]*A[6]) * invD;
    out[7] = -(A[0]*A[7]-A[1]*A[6]) * invD;
    out[8] =  (A[0]*A[4]-A[1]*A[3]) * invD;
}

inline void mul(const double* A, const double* B, double* C) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            C[i*3+j] = 0.0;
            for (int k = 0; k < 3; k++)
                C[i*3+j] += A[i*3+k] * B[k*3+j];
        }
}

inline double trace(const double* A) {
    return A[0] + A[4] + A[8];
}

/// Double contraction A:B = Σ A_ij B_ij
inline double doubleContract(const double* A, const double* B) {
    double s = 0;
    for (int i = 0; i < 9; i++) s += A[i] * B[i];
    return s;
}

} // namespace mat3util

// ─── Material Model Interface ────────────────────────────────────

/// Abstract material model for ANCF continuum elements
class MaterialModel {
public:
    virtual ~MaterialModel() = default;

    /// Strain energy density Ψ(F)
    virtual double strainEnergyDensity(const double* F) const = 0;

    /// First Piola-Kirchhoff stress P(F) → out[9]
    virtual void firstPiolaStress(const double* F, double* P) const = 0;

    /// Second Piola-Kirchhoff stress S(F) → out[9]
    virtual void secondPiolaStress(const double* F, double* S) const = 0;

    /// Material density (kg/m³)
    virtual double density() const = 0;
};

/// Compute Green-Lagrange strain E = ½(FᵀF - I)
void greenLagrangeStrain(const double* F, double* E);

// ─── St. Venant-Kirchhoff Material ──────────────────────────────

class StVenantKirchhoff : public MaterialModel {
public:
    StVenantKirchhoff(double E, double nu, double rho);

    double strainEnergyDensity(const double* F) const override;
    void firstPiolaStress(const double* F, double* P) const override;
    void secondPiolaStress(const double* F, double* S) const override;
    double density() const override { return rho_; }

private:
    double lambda_, mu_, rho_;
};

// ─── Neo-Hookean Material ───────────────────────────────────────

class NeoHookean : public MaterialModel {
public:
    NeoHookean(double E, double nu, double rho);

    double strainEnergyDensity(const double* F) const override;
    void firstPiolaStress(const double* F, double* P) const override;
    void secondPiolaStress(const double* F, double* S) const override;
    double density() const override { return rho_; }

private:
    double lambda_, mu_, rho_;
};

// ─── Factory ────────────────────────────────────────────────────

std::unique_ptr<MaterialModel> createMaterial(const ElasticMaterialProps& props);

} // namespace mb
