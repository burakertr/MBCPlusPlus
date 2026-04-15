#include "mb/core/ReferenceFrame.h"

namespace mb {

ReferenceFrame::ReferenceFrame(const Vec3& pos, const Quaternion& ori, ReferenceFrame* parent)
    : position(pos), orientation(ori), parent(parent) {}

Vec3 ReferenceFrame::pointToWorld(const Vec3& localPoint) const {
    Vec3 worldPoint = position.add(orientation.rotate(localPoint));
    if (parent) return parent->pointToWorld(worldPoint);
    return worldPoint;
}

Vec3 ReferenceFrame::pointToLocal(const Vec3& worldPoint) const {
    Vec3 p = worldPoint;
    if (parent) p = parent->pointToLocal(p);
    return orientation.inverseRotate(p.sub(position));
}

Vec3 ReferenceFrame::directionToWorld(const Vec3& localDir) const {
    Vec3 worldDir = orientation.rotate(localDir);
    if (parent) return parent->directionToWorld(worldDir);
    return worldDir;
}

Vec3 ReferenceFrame::directionToLocal(const Vec3& worldDir) const {
    Vec3 d = worldDir;
    if (parent) d = parent->directionToLocal(d);
    return orientation.inverseRotate(d);
}

Mat4 ReferenceFrame::getTransformToWorld() const {
    Mat4 T = Mat4::fromQuaternionTranslation(orientation, position);
    if (parent) return parent->getTransformToWorld().multiply(T);
    return T;
}

ReferenceFrame ReferenceFrame::compose(const ReferenceFrame& par, const ReferenceFrame& child) {
    return ReferenceFrame(
        par.position.add(par.orientation.rotate(child.position)),
        par.orientation.multiply(child.orientation)
    );
}

ReferenceFrame ReferenceFrame::inverse() const {
    Quaternion invOri = orientation.conjugate();
    Vec3 invPos = invOri.rotate(position).negate();
    return ReferenceFrame(invPos, invOri);
}

} // namespace mb
