#pragma once
#include "Force.h"
#include "mb/core/RigidBody.h"
#include <functional>

namespace mb {

/**
 * Applied force at a point on a body (constant or time-varying)
 */
class AppliedForce : public SingleBodyForce {
public:
    using ForceFunction = std::function<Vec3(double)>;

    AppliedForce(RigidBody* body, const Vec3& force, const Vec3& localPoint = Vec3::zero(),
                 const std::string& name = "AppliedForce");
    AppliedForce(RigidBody* body, ForceFunction forceFunc, const Vec3& localPoint = Vec3::zero(),
                 const std::string& name = "AppliedForce");

    void apply(double t) override;

private:
    Vec3 constantForce_;
    ForceFunction forceFunc_;
    Vec3 localPoint_;
    bool isTimeVarying_;
};

/**
 * Applied torque on a body (world or body frame)
 */
class AppliedTorque : public SingleBodyForce {
public:
    using TorqueFunction = std::function<Vec3(double)>;

    AppliedTorque(RigidBody* body, const Vec3& torque, bool inBodyFrame = false,
                  const std::string& name = "AppliedTorque");
    AppliedTorque(RigidBody* body, TorqueFunction torqueFunc, bool inBodyFrame = false,
                  const std::string& name = "AppliedTorque");

    void apply(double t) override;

private:
    Vec3 constantTorque_;
    TorqueFunction torqueFunc_;
    bool inBodyFrame_;
    bool isTimeVarying_;
};

} // namespace mb
