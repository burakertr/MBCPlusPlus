#pragma once
#include "mb/fem/ANCFTypes.h"
#include "mb/fem/ElasticMaterial.h"
#include <array>
#include <vector>

namespace mb {

/// Gauss quadrature point (tetrahedral natural coords + weight)
struct GaussPoint {
    double xi, eta, zeta;
    double weight;
};

/**
 * ANCF 4-node tetrahedral solid element (48 DOF).
 *
 * Each node has 12 DOFs: 3 position + 9 deformation gradient.
 * The element uses isoparametric linear tet shape functions
 * and Gauss quadrature for integration over the reference domain.
 */
class ANCFTetrahedralElement {
public:
    std::array<int, 4> nodeIds;
    bool alive = true;  ///< false = element has been removed (fracture)

    /**
     * Construct from connectivity, node data, and material.
     * Computes and caches reference Jacobian, volume, shape function derivatives.
     *
     * @param conn       Connectivity (4 node indices)
     * @param nodes      All ANCF nodes
     * @param material   Material model
     * @param highOrder  Use 4-point Gauss quadrature (vs 1-point)
     */
    ANCFTetrahedralElement(const TetConnectivity& conn,
                           const std::vector<ANCFNode>& nodes,
                           const MaterialModel& material,
                           bool highOrder = true);

    /// Reference volume V₀
    double V0;

    /// Compute element mass matrix (48×48, constant)
    /// Returns row-major flat array.
    std::vector<double> computeMassMatrix(const std::vector<ANCFNode>& nodes) const;

    /// Compute elastic internal forces Qₑ (48 entries)
    std::vector<double> computeElasticForces(const std::vector<ANCFNode>& nodes) const;

    /// Compute deformation gradient F at natural coords (ξ,η,ζ)
    void computeDeformationGradient(double xi, double eta, double zeta,
                                    const std::vector<ANCFNode>& nodes,
                                    double* F) const;

private:
    const MaterialModel& material_;
    std::vector<GaussPoint> gaussPoints_;

    /// Reference Jacobian J₀ and its determinant
    double J0_[9];
    double detJ0_;

    /// Pre-computed shape function derivatives in reference coords: dN/dX [4][3]
    double dNdX_[4][3];

    /// Get Gauss points for tet (1-pt or 4-pt)
    static std::vector<GaussPoint> tetGaussPoints(bool highOrder);

    /// Build 3×48 shape function matrix at (ξ,η,ζ)
    void shapeMatrix(double xi, double eta, double zeta,
                     const std::vector<ANCFNode>& nodes,
                     double* S) const;
};

} // namespace mb
