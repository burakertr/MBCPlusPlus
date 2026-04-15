#include "mb/forces/AppliedForce.h"

namespace mb {

AppliedForce::AppliedForce(RigidBody* body, const Vec3& force, const Vec3& localPoint,
                           const std::string& name)
    : SingleBodyForce(body, name), constantForce_(force), localPoint_(localPoint),
      isTimeVarying_(false) {}

AppliedForce::AppliedForce(RigidBody* body, ForceFunction forceFunc, const Vec3& localPoint,
                           const std::string& name)
    : SingleBodyForce(body, name), forceFunc_(forceFunc), localPoint_(localPoint),
      isTimeVarying_(true) {}

void AppliedForce::apply(double t) {
    Vec3 force = isTimeVarying_ ? forceFunc_(t) : constantForce_;
    Vec3 worldPoint = body_->bodyToWorld(localPoint_);
    body_->applyForceAtPoint(force, worldPoint);
}

// AppliedTorque

AppliedTorque::AppliedTorque(RigidBody* body, const Vec3& torque, bool inBodyFrame,
                             const std::string& name)
    : SingleBodyForce(body, name), constantTorque_(torque),
      inBodyFrame_(inBodyFrame), isTimeVarying_(false) {}

AppliedTorque::AppliedTorque(RigidBody* body, TorqueFunction torqueFunc, bool inBodyFrame,
                             const std::string& name)
    : SingleBodyForce(body, name), torqueFunc_(torqueFunc),
      inBodyFrame_(inBodyFrame), isTimeVarying_(true) {}

void AppliedTorque::apply(double t) {
    Vec3 torque = isTimeVarying_ ? torqueFunc_(t) : constantTorque_;
    if (inBodyFrame_) {
        torque = body_->bodyToWorldDirection(torque);
    }
    body_->applyTorque(torque);
}

} // namespace mb
