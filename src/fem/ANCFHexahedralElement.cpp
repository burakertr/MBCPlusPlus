#include "mb/fem/ANCFHexahedralElement.h"
#include <algorithm>
#include <cmath>

namespace mb {

std::vector<HexGaussPoint> ANCFHexahedralElement::hexGaussPoints(bool highOrder) {
    std::vector<HexGaussPoint> pts;
    if (!highOrder) {
        pts.push_back({0.0, 0.0, 0.0, 8.0});
        return pts;
    }

    const double a = 1.0 / std::sqrt(3.0);
    const double w = 1.0;
    for (int ix = 0; ix < 2; ix++) {
        for (int iy = 0; iy < 2; iy++) {
            for (int iz = 0; iz < 2; iz++) {
                pts.push_back({ix == 0 ? -a : a,
                               iy == 0 ? -a : a,
                               iz == 0 ? -a : a,
                               w});
            }
        }
    }
    return pts;
}

ANCFHexahedralElement::ANCFHexahedralElement(
    const HexConnectivity& conn,
    const std::vector<ANCFNode>& nodes,
    const MaterialModel& material,
    bool highOrder)
    : material_(material)
{
    nodeIds = conn.nodeIds;
    gaussPoints_ = hexGaussPoints(highOrder);

    V0 = 0.0;
    for (const auto& gp : gaussPoints_) {
        double N[8], dNdXi[8][3];
        shapeFunctions(gp.xi, gp.eta, gp.zeta, N, dNdXi);

        double J[9], detJ, Jinv[9];
        computeJacobian(nodeIds, nodes, dNdXi, J, &detJ, Jinv);
        V0 += std::abs(detJ) * gp.weight;
    }
}

void ANCFHexahedralElement::shapeFunctions(
    double xi, double eta, double zeta,
    double N[8], double dNdXi[8][3])
{
    static const int sx[8] = {-1, 1, 1, -1, -1, 1, 1, -1};
    static const int sy[8] = {-1,-1, 1,  1, -1,-1, 1,  1};
    static const int sz[8] = {-1,-1,-1, -1,  1, 1, 1,  1};

    for (int a = 0; a < 8; a++) {
        double ax = 1.0 + sx[a] * xi;
        double ay = 1.0 + sy[a] * eta;
        double az = 1.0 + sz[a] * zeta;

        N[a] = 0.125 * ax * ay * az;
        dNdXi[a][0] = 0.125 * sx[a] * ay * az;
        dNdXi[a][1] = 0.125 * sy[a] * ax * az;
        dNdXi[a][2] = 0.125 * sz[a] * ax * ay;
    }
}

void ANCFHexahedralElement::computeJacobian(
    const std::array<int, 8>& nodeIds,
    const std::vector<ANCFNode>& nodes,
    const double dNdXi[8][3],
    double* J, double* detJ, double* Jinv)
{
    std::fill(J, J + 9, 0.0);

    for (int a = 0; a < 8; a++) {
        const auto& X = nodes[nodeIds[a]].X0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                J[i*3 + j] += X[i] * dNdXi[a][j];
            }
        }
    }

    *detJ = mat3util::det(J);
    mat3util::inv(J, Jinv);
}

void ANCFHexahedralElement::mapShapeGradients(
    const double dNdXi[8][3],
    const double* Jinv,
    double dNdX[8][3])
{
    for (int a = 0; a < 8; a++) {
        for (int j = 0; j < 3; j++) {
            dNdX[a][j] = 0.0;
            for (int k = 0; k < 3; k++) {
                dNdX[a][j] += Jinv[k*3 + j] * dNdXi[a][k];
            }
        }
    }
}

std::vector<double> ANCFHexahedralElement::computeMassMatrix(const std::vector<ANCFNode>& nodes) const {
    std::vector<double> Me(ANCF_HEX_ELEM_DOF * ANCF_HEX_ELEM_DOF, 0.0);
    const double rho = material_.density();

    for (const auto& gp : gaussPoints_) {
        double N[8], dNdXi[8][3];
        shapeFunctions(gp.xi, gp.eta, gp.zeta, N, dNdXi);

        double J[9], detJ, Jinv[9];
        computeJacobian(nodeIds, nodes, dNdXi, J, &detJ, Jinv);

        const double factor = rho * std::abs(detJ) * gp.weight;
        for (int a = 0; a < 8; a++) {
            for (int b = 0; b < 8; b++) {
                const double m = factor * N[a] * N[b];
                int offA = a * ANCF_NODE_DOF;
                int offB = b * ANCF_NODE_DOF;
                for (int i = 0; i < 3; i++) {
                    Me[(offA + i) * ANCF_HEX_ELEM_DOF + (offB + i)] += m;
                }
            }
        }
    }

    return Me;
}

void ANCFHexahedralElement::computePositionBasedF(
    const std::vector<ANCFNode>& nodes,
    double* F,
    double xi,
    double eta,
    double zeta) const
{
    double N[8], dNdXi[8][3], dNdX[8][3];
    shapeFunctions(xi, eta, zeta, N, dNdXi);

    double J[9], detJ, Jinv[9];
    computeJacobian(nodeIds, nodes, dNdXi, J, &detJ, Jinv);
    mapShapeGradients(dNdXi, Jinv, dNdX);

    std::fill(F, F + 9, 0.0);
    for (int a = 0; a < 8; a++) {
        const auto& nd = nodes[nodeIds[a]];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                F[i*3 + j] += nd.q[i] * dNdX[a][j];
            }
        }
    }
}

void ANCFHexahedralElement::computeDeformationGradient(
    double xi, double eta, double zeta,
    const std::vector<ANCFNode>& nodes,
    double* F) const
{
    computePositionBasedF(nodes, F, xi, eta, zeta);
}

void ANCFHexahedralElement::computeMaterialTangent(const double* F, double* dPdF) const {
    constexpr double eps = 1e-7;
    double Fpert[9], Pplus[9], Pminus[9];

    for (int kl = 0; kl < 9; kl++) {
        for (int m = 0; m < 9; m++) Fpert[m] = F[m];
        Fpert[kl] += eps;
        material_.firstPiolaStress(Fpert, Pplus);

        for (int m = 0; m < 9; m++) Fpert[m] = F[m];
        Fpert[kl] -= eps;
        material_.firstPiolaStress(Fpert, Pminus);

        for (int ij = 0; ij < 9; ij++) {
            dPdF[ij * 9 + kl] = (Pplus[ij] - Pminus[ij]) / (2.0 * eps);
        }
    }
}

std::vector<double> ANCFHexahedralElement::computeElasticForcesPositionOnly(
    const std::vector<ANCFNode>& nodes) const
{
    std::vector<double> Qe(ANCF_HEX_ELEM_DOF, 0.0);

    for (const auto& gp : gaussPoints_) {
        double N[8], dNdXi[8][3], dNdX[8][3];
        shapeFunctions(gp.xi, gp.eta, gp.zeta, N, dNdXi);

        double J[9], detJ, Jinv[9];
        computeJacobian(nodeIds, nodes, dNdXi, J, &detJ, Jinv);
        mapShapeGradients(dNdXi, Jinv, dNdX);

        double F[9], P[9];
        std::fill(F, F + 9, 0.0);
        for (int a = 0; a < 8; a++) {
            const auto& nd = nodes[nodeIds[a]];
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    F[i*3 + j] += nd.q[i] * dNdX[a][j];
                }
            }
        }

        material_.firstPiolaStress(F, P);
        const double factor = std::abs(detJ) * gp.weight;

        for (int a = 0; a < 8; a++) {
            int off = a * ANCF_NODE_DOF;
            for (int k = 0; k < 3; k++) {
                double v = 0.0;
                for (int j = 0; j < 3; j++) {
                    v += P[k*3 + j] * dNdX[a][j];
                }
                Qe[off + k] -= v * factor;
            }
        }
    }

    return Qe;
}

std::vector<double> ANCFHexahedralElement::computeStiffnessMatrixPositionOnly(
    const std::vector<ANCFNode>& nodes) const
{
    std::vector<double> Ke(ANCF_HEX_ELEM_DOF * ANCF_HEX_ELEM_DOF, 0.0);

    for (const auto& gp : gaussPoints_) {
        double N[8], dNdXi[8][3], dNdX[8][3];
        shapeFunctions(gp.xi, gp.eta, gp.zeta, N, dNdXi);

        double J[9], detJ, Jinv[9];
        computeJacobian(nodeIds, nodes, dNdXi, J, &detJ, Jinv);
        mapShapeGradients(dNdXi, Jinv, dNdX);

        double F[9];
        std::fill(F, F + 9, 0.0);
        for (int a = 0; a < 8; a++) {
            const auto& nd = nodes[nodeIds[a]];
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    F[i*3 + j] += nd.q[i] * dNdX[a][j];
                }
            }
        }

        double dPdF[81];
        computeMaterialTangent(F, dPdF);

        const double factor = std::abs(detJ) * gp.weight;
        for (int a = 0; a < 8; a++) {
            for (int b = 0; b < 8; b++) {
                int offA = a * ANCF_NODE_DOF;
                int offB = b * ANCF_NODE_DOF;
                for (int m1 = 0; m1 < 3; m1++) {
                    for (int m2 = 0; m2 < 3; m2++) {
                        double v = 0.0;
                        for (int j = 0; j < 3; j++) {
                            for (int l = 0; l < 3; l++) {
                                v += dNdX[a][j] * dPdF[(m1*3 + j) * 9 + (m2*3 + l)] * dNdX[b][l];
                            }
                        }
                        Ke[(offA + m1) * ANCF_HEX_ELEM_DOF + (offB + m2)] += v * factor;
                    }
                }
            }
        }
    }

    return Ke;
}

std::vector<double> ANCFHexahedralElement::computeElasticForces(
    const std::vector<ANCFNode>& nodes) const
{
    return computeElasticForcesPositionOnly(nodes);
}

std::vector<double> ANCFHexahedralElement::computeStiffnessMatrix(
    const std::vector<ANCFNode>& nodes) const
{
    return computeStiffnessMatrixPositionOnly(nodes);
}

} // namespace mb
