#pragma once
#include <vector>
#include <array>
#include <cmath>
#include <string>

namespace mb {

/// ANCF node DOF count: 3 position + 9 deformation gradient = 12
constexpr int ANCF_NODE_DOF = 12;

/// ANCF element DOF count: 4 nodes × 12 = 48
constexpr int ANCF_ELEM_DOF = 48;

/// ANCF hexahedral element DOF count: 8 nodes × 12 = 96
constexpr int ANCF_HEX_ELEM_DOF = 96;

/// ANCF node:
///   q[0..2]  = position (rx, ry, rz)
///   q[3..5]  = F column 0 (F11, F21, F31)
///   q[6..8]  = F column 1 (F12, F22, F32)
///   q[9..11] = F column 2 (F13, F23, F33)
struct ANCFNode {
    int id;
    std::array<double, 3> X0;                       ///< Reference position
    std::array<double, ANCF_NODE_DOF> q;             ///< Generalized coordinates
    std::array<double, ANCF_NODE_DOF> qd;            ///< Generalized velocities
    std::array<double, ANCF_NODE_DOF> qdd;           ///< Generalized accelerations
    bool fixed = false;
    std::array<bool, ANCF_NODE_DOF> fixedDOF = {};   ///< Per-DOF BC
};

/// Tetrahedral element connectivity (4 node indices)
struct TetConnectivity {
    std::array<int, 4> nodeIds;
};

/// Hexahedral element connectivity (8 node indices)
struct HexConnectivity {
    std::array<int, 8> nodeIds;
};

/// Material type enumeration
enum class MaterialType {
    StVenantKirchhoff,
    NeoHookean
};

/// Elastic material properties
struct ElasticMaterialProps {
    double E;                ///< Young's modulus (Pa)
    double nu;               ///< Poisson's ratio
    double rho;              ///< Density (kg/m³)
    MaterialType type = MaterialType::StVenantKirchhoff;
};

/// Lamé parameters
struct LameParams {
    double lambda;
    double mu;
};

/// Compute Lamé parameters from E and ν
inline LameParams computeLame(double E, double nu) {
    return {
        E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)),   // λ
        E / (2.0 * (1.0 + nu))                         // μ
    };
}

/// Gmsh node
struct GmshNode {
    int id;
    double x, y, z;
};

/// Gmsh element
struct GmshElement {
    int id;
    int type;
    std::vector<int> nodeIds;
};

/// Gmsh mesh (nodes + tet4/hex8 elements)
struct GmshMesh {
    std::vector<GmshNode> nodes;
    std::vector<GmshElement> elements;
};

} // namespace mb
