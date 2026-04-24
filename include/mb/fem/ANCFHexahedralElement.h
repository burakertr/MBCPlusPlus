#pragma once
#include "mb/fem/ANCFTypes.h"
#include "mb/fem/ElasticMaterial.h"
#include <array>
#include <vector>

namespace mb {

struct HexGaussPoint {
    double xi, eta, zeta;
    double weight;
};

class ANCFHexahedralElement {
public:
    std::array<int, 8> nodeIds;
    bool alive = true;
    double V0 = 0.0;

    ANCFHexahedralElement(const HexConnectivity& conn,
                          const std::vector<ANCFNode>& nodes,
                          const MaterialModel& material,
                          bool highOrder = true);

    std::vector<double> computeMassMatrix(const std::vector<ANCFNode>& nodes) const;
    std::vector<double> computeElasticForces(const std::vector<ANCFNode>& nodes) const;
    std::vector<double> computeStiffnessMatrix(const std::vector<ANCFNode>& nodes) const;

    std::vector<double> computeElasticForcesPositionOnly(const std::vector<ANCFNode>& nodes) const;
    std::vector<double> computeStiffnessMatrixPositionOnly(const std::vector<ANCFNode>& nodes) const;

    void computeDeformationGradient(double xi, double eta, double zeta,
                                    const std::vector<ANCFNode>& nodes,
                                    double* F) const;

    void computePositionBasedF(const std::vector<ANCFNode>& nodes,
                               double* F,
                               double xi = 0.0,
                               double eta = 0.0,
                               double zeta = 0.0) const;

    void computeMaterialTangent(const double* F, double* dPdF) const;

private:
    const MaterialModel& material_;
    std::vector<HexGaussPoint> gaussPoints_;

    static std::vector<HexGaussPoint> hexGaussPoints(bool highOrder);

    static void shapeFunctions(double xi, double eta, double zeta,
                               double N[8], double dNdXi[8][3]);

    static void computeJacobian(const std::array<int, 8>& nodeIds,
                                const std::vector<ANCFNode>& nodes,
                                const double dNdXi[8][3],
                                double* J, double* detJ, double* Jinv);

    static void mapShapeGradients(const double dNdXi[8][3],
                                  const double* Jinv,
                                  double dNdX[8][3]);
};

} // namespace mb
