#pragma once
#include "mb/math/Vec3.h"
#include "mb/math/Quaternion.h"
#include "mb/math/Mat4.h"

namespace mb {

/**
 * Reference frame for coordinate transformations
 */
class ReferenceFrame {
public:
    Vec3 position;
    Quaternion orientation;
    ReferenceFrame* parent;

    ReferenceFrame(const Vec3& pos = Vec3::zero(),
                   const Quaternion& ori = Quaternion::identity(),
                   ReferenceFrame* parent = nullptr);

    // Transform point from this frame to world
    Vec3 pointToWorld(const Vec3& localPoint) const;
    // Transform point from world to this frame
    Vec3 pointToLocal(const Vec3& worldPoint) const;
    // Transform direction from this frame to world
    Vec3 directionToWorld(const Vec3& localDir) const;
    // Transform direction from world to this frame
    Vec3 directionToLocal(const Vec3& worldDir) const;

    // Get 4x4 transformation matrix (this frame → world)
    Mat4 getTransformToWorld() const;

    // Compose two frames
    static ReferenceFrame compose(const ReferenceFrame& parent, const ReferenceFrame& child);
    // Inverse frame
    ReferenceFrame inverse() const;
};

} // namespace mb
