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
 *
 * Full ANCF formulation:
 *   Position field:  r(X) = Σ_a Na(X) · ra
 *   Deformation gradient:  F(X) = Σ_a Na(X) · Fa  (gradient DOFs)
 *
 * The element uses isoparametric linear tet shape functions
 * and Gauss quadrature for integration over the reference domain.
 *
 * Elastic forces act on ALL 48 DOFs (position + gradient).
 * An internal constraint couples position-derived F with gradient DOFs
 * for consistency.
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
    /// Full ANCF: forces on both position and gradient DOFs
    std::vector<double> computeElasticForces(const std::vector<ANCFNode>& nodes) const;

    /// Compute element tangent stiffness matrix Kₑ (48×48)
    /// Returns row-major flat array.
    /// Uses analytic differentiation of the first Piola-Kirchhoff stress.
    std::vector<double> computeStiffnessMatrix(const std::vector<ANCFNode>& nodes) const;

    /// Compute deformation gradient F at natural coords (ξ,η,ζ)
    /// Full ANCF: interpolates gradient DOFs directly
    void computeDeformationGradient(double xi, double eta, double zeta,
                                    const std::vector<ANCFNode>& nodes,
                                    double* F) const;

    /// Compute deformation gradient from position DOFs only (∂r/∂X)
    void computePositionBasedF(const std::vector<ANCFNode>& nodes,
                               double* F) const;

    /// Compute F-bar (volumetric-locking-free deformation gradient).
    /// Replaces volumetric part of F with averaged Jacobian:
    ///   F̄ = (J̄/J)^{1/3} · F
    /// @param Javg  Volume-weighted average Jacobian from surrounding patch
    void computeFbar(const std::vector<ANCFNode>& nodes,
                     double Javg, double* Fbar) const;

    /// Pre-computed shape function derivatives in reference coords: dN/dX [4][3]
    double dNdX_[4][3];

    /// Compute material tangent ∂P/∂F numerically (9×9 Voigt → full tensor)
    void computeMaterialTangent(const double* F, double* dPdF) const;

private:
    const MaterialModel& material_;
    std::vector<GaussPoint> gaussPoints_;

    /// Reference Jacobian J₀ and its determinant
    double J0_[9];
    double detJ0_;

    /// Get Gauss points for tet (1-pt or 4-pt)
    static std::vector<GaussPoint> tetGaussPoints(bool highOrder);

    /// Build 3×48 shape function matrix at (ξ,η,ζ)
    void shapeMatrix(double xi, double eta, double zeta,
                     const std::vector<ANCFNode>& nodes,
                     double* S) const;
};

} // namespace mb
