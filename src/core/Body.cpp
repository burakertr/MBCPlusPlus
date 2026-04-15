#include "mb/core/Body.h"

namespace mb {

int Body::nextId_ = 0;

Body::Body(BodyType type, const std::string& name)
    : id(nextId_++), name(name), type(type),
      accumulatedForce(Vec3::zero()), accumulatedTorque(Vec3::zero()) {
    if (this->name.empty()) this->name = "Body_" + std::to_string(id);
}

void Body::clearForces() {
    accumulatedForce = Vec3::zero();
    accumulatedTorque = Vec3::zero();
}

void Body::applyForce(const Vec3& force) {
    accumulatedForce = accumulatedForce.add(force);
}

void Body::applyTorque(const Vec3& torque) {
    accumulatedTorque = accumulatedTorque.add(torque);
}

void Body::applyForceAtPoint(const Vec3& force, const Vec3& worldPoint) {
    accumulatedForce = accumulatedForce.add(force);
    Vec3 r = worldPoint.sub(position);
    accumulatedTorque = accumulatedTorque.add(r.cross(force));
}

Vec3 Body::bodyToWorld(const Vec3& localPoint) const {
    return position.add(orientation.rotate(localPoint));
}

Vec3 Body::worldToBody(const Vec3& worldPoint) const {
    return orientation.inverseRotate(worldPoint.sub(position));
}

Vec3 Body::bodyToWorldDirection(const Vec3& localDir) const {
    return orientation.rotate(localDir);
}

Vec3 Body::worldToBodyDirection(const Vec3& worldDir) const {
    return orientation.inverseRotate(worldDir);
}

Vec3 Body::getPointVelocity(const Vec3& worldPoint) const {
    Vec3 r = worldPoint.sub(position);
    return velocity.add(angularVelocity.cross(r));
}

} // namespace mb
