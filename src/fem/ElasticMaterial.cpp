#include "mb/fem/ElasticMaterial.h"
#include <cmath>

namespace mb {

// ─── Green-Lagrange Strain ───────────────────────────────────

void greenLagrangeStrain(const double* F, double* E) {
    // E = 0.5 * (F^T F - I)
    // F is 3×3 row-major
    double FtF[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            FtF[i*3+j] = 0;
            for (int k = 0; k < 3; k++)
                FtF[i*3+j] += F[k*3+i] * F[k*3+j];
        }
    for (int i = 0; i < 9; i++)
        E[i] = 0.5 * FtF[i];
    E[0] -= 0.5; E[4] -= 0.5; E[8] -= 0.5;
}

// ─── St. Venant-Kirchhoff ───────────────────────────────────

StVenantKirchhoff::StVenantKirchhoff(double E, double nu, double rho) : rho_(rho) {
    auto lame = computeLame(E, nu);
    lambda_ = lame.lambda;
    mu_ = lame.mu;
}

double StVenantKirchhoff::strainEnergyDensity(const double* F) const {
    double E[9];
    greenLagrangeStrain(F, E);
    double trE = E[0] + E[4] + E[8];
    double EE = mat3util::doubleContract(E, E);
    return 0.5 * lambda_ * trE * trE + mu_ * EE;
}

void StVenantKirchhoff::secondPiolaStress(const double* F, double* S) const {
    double E[9];
    greenLagrangeStrain(F, E);
    double trE = E[0] + E[4] + E[8];
    // S = λ tr(E) I + 2μ E
    for (int i = 0; i < 9; i++)
        S[i] = 2.0 * mu_ * E[i];
    S[0] += lambda_ * trE;
    S[4] += lambda_ * trE;
    S[8] += lambda_ * trE;
}

void StVenantKirchhoff::firstPiolaStress(const double* F, double* P) const {
    double S[9];
    secondPiolaStress(F, S);
    // P = F S
    mat3util::mul(F, S, P);
}

// ─── Neo-Hookean ─────────────────────────────────────────────

NeoHookean::NeoHookean(double E, double nu, double rho) : rho_(rho) {
    auto lame = computeLame(E, nu);
    lambda_ = lame.lambda;
    mu_ = lame.mu;
}

double NeoHookean::strainEnergyDensity(const double* F) const {
    // C = F^T F
    double C[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            C[i*3+j] = 0;
            for (int k = 0; k < 3; k++)
                C[i*3+j] += F[k*3+i] * F[k*3+j];
        }
    double I1 = C[0] + C[4] + C[8];
    double J = mat3util::det(F);
    if (J < 1e-20) J = 1e-20;
    double lnJ = std::log(J);
    return 0.5 * mu_ * (I1 - 3.0) - mu_ * lnJ + 0.5 * lambda_ * lnJ * lnJ;
}

void NeoHookean::secondPiolaStress(const double* F, double* S) const {
    // C = F^T F, then C^{-1}
    double C[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            C[i*3+j] = 0;
            for (int k = 0; k < 3; k++)
                C[i*3+j] += F[k*3+i] * F[k*3+j];
        }
    double Cinv[9];
    mat3util::inv(C, Cinv);

    double J = mat3util::det(F);
    if (std::abs(J) < 1e-20) J = 1e-20;
    double lnJ = std::log(std::abs(J));

    // S = μ(I - C^{-1}) + λ lnJ C^{-1}
    for (int i = 0; i < 9; i++)
        S[i] = -mu_ * Cinv[i] + lambda_ * lnJ * Cinv[i];
    S[0] += mu_;
    S[4] += mu_;
    S[8] += mu_;
}

void NeoHookean::firstPiolaStress(const double* F, double* P) const {
    // Direct computation of P for Neo-Hookean:
    // P = μ·F + (λ·ln(J) - μ)·F^{-T}
    double J = mat3util::det(F);
    if (std::abs(J) < 1e-20) J = (J >= 0) ? 1e-20 : -1e-20;
    double lnJ = std::log(std::abs(J));

    // F^{-T} = (F^{-1})^T
    double Finv[9];
    mat3util::inv(F, Finv);

    double coeff = lambda_ * lnJ - mu_;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            P[i*3+j] = mu_ * F[i*3+j] + coeff * Finv[j*3+i];  // Finv^T_{ij} = Finv_{ji}
}

// ─── Factory ─────────────────────────────────────────────────

std::unique_ptr<MaterialModel> createMaterial(const ElasticMaterialProps& props) {
    switch (props.type) {
        case MaterialType::NeoHookean:
            return std::make_unique<NeoHookean>(props.E, props.nu, props.rho);
        case MaterialType::StVenantKirchhoff:
        default:
            return std::make_unique<StVenantKirchhoff>(props.E, props.nu, props.rho);
    }
}

} // namespace mb
