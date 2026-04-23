#include "mb/fem/ANCFTetrahedralElement.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace mb {

// ─── Gauss Quadrature for Tetrahedra ─────────────────────────

std::vector<GaussPoint> ANCFTetrahedralElement::tetGaussPoints(bool highOrder) {
    std::vector<GaussPoint> pts;
    if (!highOrder) {
        // 1-point rule (centroid)
        pts.push_back({0.25, 0.25, 0.25, 1.0/6.0});
    } else {
        // 4-point rule
        double a = 0.1381966011250105;
        double b = 0.5854101966249685;
        double w = 1.0/24.0;
        pts.push_back({a, a, a, w});
        pts.push_back({b, a, a, w});
        pts.push_back({a, b, a, w});
        pts.push_back({a, a, b, w});
    }
    return pts;
}

// ─── Local 3×3 helpers ───────────────────────────────────────

static double det3(const double* A) {
    return A[0]*(A[4]*A[8]-A[5]*A[7])
         - A[1]*(A[3]*A[8]-A[5]*A[6])
         + A[2]*(A[3]*A[7]-A[4]*A[6]);
}

static void inv3(const double* A, double* out) {
    double d = det3(A);
    if (std::abs(d) < 1e-30) d = 1e-30;
    double id = 1.0/d;
    out[0] =  (A[4]*A[8]-A[5]*A[7])*id;
    out[1] = -(A[1]*A[8]-A[2]*A[7])*id;
    out[2] =  (A[1]*A[5]-A[2]*A[4])*id;
    out[3] = -(A[3]*A[8]-A[5]*A[6])*id;
    out[4] =  (A[0]*A[8]-A[2]*A[6])*id;
    out[5] = -(A[0]*A[5]-A[2]*A[3])*id;
    out[6] =  (A[3]*A[7]-A[4]*A[6])*id;
    out[7] = -(A[0]*A[7]-A[1]*A[6])*id;
    out[8] =  (A[0]*A[4]-A[1]*A[3])*id;
}

// ─── Constructor ─────────────────────────────────────────────

ANCFTetrahedralElement::ANCFTetrahedralElement(
    const TetConnectivity& conn,
    const std::vector<ANCFNode>& nodes,
    const MaterialModel& material,
    bool highOrder)
    : material_(material)
{
    nodeIds = conn.nodeIds;
    gaussPoints_ = tetGaussPoints(highOrder);

    // Reference Jacobian: J0 = [X1-X0 | X2-X0 | X3-X0]^T  (row-major)
    const auto& n0 = nodes[nodeIds[0]].X0;
    const auto& n1 = nodes[nodeIds[1]].X0;
    const auto& n2 = nodes[nodeIds[2]].X0;
    const auto& n3 = nodes[nodeIds[3]].X0;

    // J0 rows = edge vectors
    J0_[0] = n1[0]-n0[0]; J0_[1] = n1[1]-n0[1]; J0_[2] = n1[2]-n0[2];
    J0_[3] = n2[0]-n0[0]; J0_[4] = n2[1]-n0[1]; J0_[5] = n2[2]-n0[2];
    J0_[6] = n3[0]-n0[0]; J0_[7] = n3[1]-n0[1]; J0_[8] = n3[2]-n0[2];

    detJ0_ = det3(J0_);

    // Auto-fix inverted element (swap nodes 2 and 3)
    if (detJ0_ < 0) {
        std::swap(nodeIds[2], nodeIds[3]);
        const auto& nn2 = nodes[nodeIds[2]].X0;
        const auto& nn3 = nodes[nodeIds[3]].X0;
        J0_[3] = nn2[0]-n0[0]; J0_[4] = nn2[1]-n0[1]; J0_[5] = nn2[2]-n0[2];
        J0_[6] = nn3[0]-n0[0]; J0_[7] = nn3[1]-n0[1]; J0_[8] = nn3[2]-n0[2];
        detJ0_ = det3(J0_);
    }

    V0 = std::abs(detJ0_) / 6.0;

    // Pre-compute dN/dX = dN/dξ · J0^{-1}
    // Shape functions: N0 = 1-ξ-η-ζ, N1=ξ, N2=η, N3=ζ
    double J0inv[9];
    inv3(J0_, J0inv);

    double dNdXi[4][3] = {
        {-1, -1, -1},
        { 1,  0,  0},
        { 0,  1,  0},
        { 0,  0,  1}
    };

    for (int a = 0; a < 4; a++)
        for (int j = 0; j < 3; j++) {
            dNdX_[a][j] = 0;
            for (int k = 0; k < 3; k++)
                dNdX_[a][j] += dNdXi[a][k] * J0inv[j*3+k];
        }
}

// ─── Shape Function Matrix ───────────────────────────────────

void ANCFTetrahedralElement::shapeMatrix(
    double xi, double eta, double zeta,
    const std::vector<ANCFNode>& nodes,
    double* S) const
{
    // S is 3×48, row-major
    // Full ANCF interpolation:
    //   r_i(X) = Σ_a Na(ξ) · [ra_i + Σ_l (X_l(ξ) - Xa_l) · Fa_il]
    // where Fa_il = q_a[3 + l*3 + i] (gradient DOF at node a)

    double N[4] = {1.0 - xi - eta - zeta, xi, eta, zeta};

    std::fill(S, S + 3*48, 0.0);

    for (int a = 0; a < 4; a++) {
        const auto& nd = nodes[nodeIds[a]];
        int colOff = a * ANCF_NODE_DOF;

        // Position contribution: Na * δ_ij → S[i][colOff + i] += Na
        for (int i = 0; i < 3; i++) {
            S[i * 48 + colOff + i] += N[a];
        }

        // Gradient contribution: Na * (X_l - Xa_l) * δ_ij for gradient DOFs
        for (int l = 0; l < 3; l++) {
            double dX_l = 0;
            for (int b = 0; b < 4; b++) {
                double Nb = (b == 0) ? (1.0-xi-eta-zeta) : (b == 1) ? xi : (b == 2) ? eta : zeta;
                dX_l += Nb * (nodes[nodeIds[b]].X0[l] - nd.X0[l]);
            }

            for (int i = 0; i < 3; i++) {
                int gradDof = 3 + l * 3 + i;
                S[i * 48 + colOff + gradDof] += N[a] * dX_l;
            }
        }
    }
}

// ─── Mass Matrix ─────────────────────────────────────────────

std::vector<double> ANCFTetrahedralElement::computeMassMatrix(
    const std::vector<ANCFNode>& nodes) const
{
    int n = ANCF_ELEM_DOF;
    std::vector<double> Me(n * n, 0.0);
    double rho = material_.density();

    // Use 4-point Gauss quadrature for mass
    auto gp4 = tetGaussPoints(true);

    for (const auto& gp : gp4) {
        double S[3 * 48];
        shapeMatrix(gp.xi, gp.eta, gp.zeta, nodes, S);

        // Me += ρ |J0| S^T S * weight
        double factor = rho * std::abs(detJ0_) * gp.weight;

        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double val = 0;
                for (int k = 0; k < 3; k++)
                    val += S[k * 48 + i] * S[k * 48 + j];
                val *= factor;
                Me[i * n + j] += val;
                if (i != j) Me[j * n + i] += val;
            }
        }
    }

    return Me;
}

// ─── Deformation Gradient ────────────────────────────────────

void ANCFTetrahedralElement::computeDeformationGradient(
    double /*xi*/, double /*eta*/, double /*zeta*/,
    const std::vector<ANCFNode>& nodes,
    double* F) const
{
    // F_ij = Σ_a (∂Na/∂Xj) · q_a_i  (position DOFs, constant over element)
    std::fill(F, F + 9, 0.0);
    for (int a = 0; a < 4; a++) {
        const auto& nd = nodes[nodeIds[a]];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                F[i*3+j] += dNdX_[a][j] * nd.q[i];
    }
}

void ANCFTetrahedralElement::computePositionBasedF(
    const std::vector<ANCFNode>& nodes,
    double* F) const
{
    computeDeformationGradient(0, 0, 0, nodes, F);
}

void ANCFTetrahedralElement::computeFbar(
    const std::vector<ANCFNode>& nodes,
    double Javg, double* Fbar) const
{
    // Standard F from position DOFs
    double F[9];
    computePositionBasedF(nodes, F);

    // J = det(F)
    double J = F[0]*(F[4]*F[8]-F[5]*F[7])
             - F[1]*(F[3]*F[8]-F[5]*F[6])
             + F[2]*(F[3]*F[7]-F[4]*F[6]);

    if (std::abs(J) < 1e-30) J = 1e-30;

    // F-bar = (Javg/J)^{1/3} * F
    double scale = std::cbrt(Javg / J);
    for (int i = 0; i < 9; i++)
        Fbar[i] = scale * F[i];
}

// ─── Material Tangent ────────────────────────────────────────

void ANCFTetrahedralElement::computeMaterialTangent(
    const double* F, double* dPdF) const
{
    // Compute ∂P_ij/∂F_kl numerically using central finite differences.
    // dPdF is stored as a 9×9 matrix: dPdF[(i*3+j)*9 + (k*3+l)] = ∂P_ij/∂F_kl

    constexpr double eps = 1e-7;
    double Fpert[9], Pplus[9], Pminus[9];

    for (int kl = 0; kl < 9; kl++) {
        // Perturb F[kl] by +eps
        for (int m = 0; m < 9; m++) Fpert[m] = F[m];
        Fpert[kl] += eps;
        material_.firstPiolaStress(Fpert, Pplus);

        // Perturb F[kl] by -eps
        for (int m = 0; m < 9; m++) Fpert[m] = F[m];
        Fpert[kl] -= eps;
        material_.firstPiolaStress(Fpert, Pminus);

        // Central difference
        for (int ij = 0; ij < 9; ij++) {
            dPdF[ij * 9 + kl] = (Pplus[ij] - Pminus[ij]) / (2.0 * eps);
        }
    }
}

// ─── Elastic Forces ──────────────────────────────────────────

std::vector<double> ANCFTetrahedralElement::computeElasticForces(
    const std::vector<ANCFNode>& nodes) const
{
    std::vector<double> Qe(ANCF_ELEM_DOF, 0.0);

    for (const auto& gp : gaussPoints_) {
        double F[9];
        computeDeformationGradient(gp.xi, gp.eta, gp.zeta, nodes, F);

        double P[9];
        material_.firstPiolaStress(F, P);

        double factor = std::abs(detJ0_) * gp.weight;

        for (int a = 0; a < 4; a++) {
            int off = a * ANCF_NODE_DOF;
            // Position DOF forces: Qe_a_k = -Σ_j P_kj · dNa/dXj · |J0| · w
            for (int k = 0; k < 3; k++) {
                double val = 0;
                for (int j = 0; j < 3; j++)
                    val += P[k*3+j] * dNdX_[a][j];
                Qe[off + k] -= val * factor;
            }
        }
    }

    return Qe;
}

// ─── Tangent Stiffness Matrix ────────────────────────────────

std::vector<double> ANCFTetrahedralElement::computeStiffnessMatrix(
    const std::vector<ANCFNode>& nodes) const
{
    int n = ANCF_ELEM_DOF;  // 48
    std::vector<double> Ke(n * n, 0.0);

    for (const auto& gp : gaussPoints_) {
        double F[9];
        computeDeformationGradient(gp.xi, gp.eta, gp.zeta, nodes, F);

        double dPdF[81];
        computeMaterialTangent(F, dPdF);

        double factor = std::abs(detJ0_) * gp.weight;

        for (int a = 0; a < 4; a++) {
            for (int b = 0; b < 4; b++) {
                int offA = a * ANCF_NODE_DOF;
                int offB = b * ANCF_NODE_DOF;

                // Position-Position block: Kpp
                for (int m1 = 0; m1 < 3; m1++) {
                    for (int m2 = 0; m2 < 3; m2++) {
                        double val = 0;
                        for (int j = 0; j < 3; j++)
                            for (int l = 0; l < 3; l++)
                                val += dNdX_[a][j] * dPdF[(m1*3+j)*9 + (m2*3+l)] * dNdX_[b][l];
                        Ke[(offA+m1)*n + (offB+m2)] += val * factor;
                    }
                }
            }
        }
    }

    return Ke;
}

} // namespace mb
