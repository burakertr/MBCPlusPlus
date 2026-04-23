#pragma once
#include "mb/core/Body.h"
#include "mb/fem/ANCFTypes.h"
#include "mb/fem/ANCFTetrahedralElement.h"
#include "mb/fem/ElasticMaterial.h"
#include <vector>
#include <memory>
#include <functional>

namespace mb {

/**
 * ANCF-based flexible body.
 *
 * Manages ANCF nodes and tetrahedral elements, assembles the global
 * mass matrix, computes elastic/gravity/total forces, and exposes
 * the Body virtual interface so it can participate in MultibodySystem.
 */
class FlexibleBody : public Body {
public:
    std::vector<ANCFNode> nodes;
    std::vector<ANCFTetrahedralElement> elements;
    std::unique_ptr<MaterialModel> material;
    ElasticMaterialProps materialProps;
    int numDof = 0;

    /// Gravity vector (can be overridden by system)
    Vec3 gravity{0, -9.81, 0};

    /// Rayleigh damping: C = αM (mass-proportional)
    double dampingAlpha = 0.0;

    /// Gradient-DOF penalty: pulls F back toward nearest rotation
    double gradientPenalty = 0.0;

    /// External force callback
    std::function<std::vector<double>(FlexibleBody&)> externalForces;

    FlexibleBody(const std::string& name = "");

    // ─── Construction ────────────────────────────────────────

    /// Build from mesh and material
    static std::shared_ptr<FlexibleBody> fromMesh(
        const GmshMesh& mesh,
        const ElasticMaterialProps& matProps,
        const std::string& name = "",
        bool highOrderQuad = true);

    // ─── Boundary Conditions ─────────────────────────────────

    void fixNode(int nodeIdx);
    void fixNodeDOFs(int nodeIdx, const std::vector<int>& dofs);
    void fixNodesOnPlane(char axis, double value, double tol = 1e-6);

    int numFreeDof() const { return (int)freeDofMap_.size(); }
    bool isDofFixed(int globalDof) const;

    // ─── State Management ────────────────────────────────────

    std::vector<double> getFlexQ() const;
    void setFlexQ(const std::vector<double>& q);
    std::vector<double> getFlexQd() const;
    void setFlexQd(const std::vector<double>& qd);

    void setAngularVelocityFlex(const Vec3& omega);
    void setLinearVelocityFlex(const Vec3& v);

    // ─── Mass Matrix ─────────────────────────────────────────

    const std::vector<double>& getGlobalMassMatrix();
    std::vector<double> getMassDiagonal();
    std::vector<double> getMassDiagonalInverse();

    // ─── Force Computation ───────────────────────────────────

    std::vector<double> computeElasticForces();
    std::vector<double> computeGravityForces();
    std::vector<double> computeTotalForces();

    // ─── Stiffness Matrix (for implicit integrators) ─────────

    /// Assemble global tangent stiffness matrix K (numDof×numDof, row-major)
    std::vector<double> assembleStiffnessMatrix();

    // ─── Analysis ────────────────────────────────────────────

    double computeStrainEnergy() const;
    double getTotalMass() const;
    double getMaxDisplacement() const;
    std::vector<Vec3> getNodePositions() const;
    std::vector<Vec3> getNodeDisplacements() const;
    std::vector<std::array<int,4>> getTetConnectivity() const;

    /// Compute von Mises stress at element centroid (Pa)
    double computeElementVonMises(int elemIdx) const;

    // ─── Element Removal (Fracture) ─────────────────────────

    /// Remove elements by index. Indices need not be sorted.
    /// Invalidates mass matrix cache automatically.
    void removeElements(const std::vector<int>& elemIndices);

    /// Invalidate cached mass matrix (call after topology changes)
    void invalidateMassCache();

    // ─── Body Interface Overrides ────────────────────────────

    double getMass() const override { return getTotalMass(); }
    int nq() const override { return numDof; }
    int nv() const override { return numDof; }

    std::vector<double> getQ() const override;
    void setQ(const std::vector<double>& q) override;
    std::vector<double> getV() const override;
    void setV(const std::vector<double>& v) override;

    std::vector<double> computeQDot() const override;
    std::vector<double> computeMassBlock() const override;
    std::vector<double> computeForces(const Vec3& gravity) override;

    double computeKineticEnergy() const override;
    double computePotentialEnergy(const Vec3& gravity) const override;

private:
    void rebuildDofMap();

    std::vector<int> freeDofMap_;
    std::vector<bool> fixedDofMask_;

    // Caches (mass matrix is constant for ANCF)
    std::vector<double> globalMass_;
    std::vector<double> massDiag_;
    std::vector<double> massDiagInv_;

    static int flexNextId_;
};

} // namespace mb
