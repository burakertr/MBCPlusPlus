#pragma once
#include "mb/math/Vec3.h"
#include "mb/fem/FlexibleBody.h"
#include <vector>
#include <set>

namespace mb {

/// A surface triangle defined by three node indices
struct SurfaceTriangle {
    int n0, n1, n2;
};

/// A detected contact between an ANCF node and a surface
struct FlexContact {
    FlexibleBody* nodeBody;
    int nodeIdx;
    FlexibleBody* surfBody;     ///< nullptr for ground
    int triIdx;                  ///< -1 for ground
    Vec3 point;                  ///< Contact point on surface
    Vec3 normal;                 ///< Outward normal
    double depth;                ///< Penetration depth (positive = overlap)
};

/// Ground plane
struct GroundPlane {
    double y;
    Vec3 normal;  ///< unit normal
};

/// Axis-aligned bounding box
struct AABB {
    double minX, minY, minZ;
    double maxX, maxY, maxZ;
};

// ─── Surface extraction ──────────────────────────────────────

/// Extract boundary triangles from a tet mesh
std::vector<SurfaceTriangle> extractSurfaceTriangles(FlexibleBody& body);

/// Get the set of node indices on the surface
std::set<int> extractSurfaceNodeIndices(const std::vector<SurfaceTriangle>& tris);

// ─── AABB helpers ────────────────────────────────────────────

AABB computeBodyAABB(const FlexibleBody& body, const std::set<int>& nodeIndices);
bool aabbOverlap(const AABB& a, const AABB& b, double margin);

// ─── Detector ────────────────────────────────────────────────

class FlexibleContactDetector {
public:
    double minDepth = 1e-8;
    double maxDepth = 0.01;
    double contactMargin = 0.005;

    /// Detect node-to-surface contacts (A nodes → B surface)
    std::vector<FlexContact> detectNodeToSurface(
        FlexibleBody& nodeBody,
        const std::set<int>& nodeIndices,
        FlexibleBody& surfBody,
        const std::vector<SurfaceTriangle>& surfTris);

    /// Detect node-to-ground contacts
    std::vector<FlexContact> detectNodeToGround(
        FlexibleBody& body,
        const std::set<int>& nodeIndices,
        const GroundPlane& ground);

    /// SAT convex-convex collision
    std::vector<FlexContact> detectSAT(
        FlexibleBody& bodyA,
        const std::vector<SurfaceTriangle>& trisA,
        FlexibleBody& bodyB,
        const std::vector<SurfaceTriangle>& trisB);

private:
    std::vector<double> triPlaneData_;  ///< Reusable buffer
};

} // namespace mb
