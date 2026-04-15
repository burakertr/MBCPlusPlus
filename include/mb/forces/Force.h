#pragma once
#include "mb/math/Vec3.h"
#include "mb/core/Body.h"
#include "mb/core/RigidBody.h"
#include <string>

namespace mb {

struct ForceResult {
    Vec3 force;
    Vec3 torque;
};

/**
 * Abstract base class for forces
 */
class Force {
public:
    std::string name;

    Force(const std::string& name = "Force") : name(name) {}
    virtual ~Force() = default;

    virtual void apply(double t) = 0;
};

/**
 * Force acting on a single body
 */
class SingleBodyForce : public Force {
public:
    SingleBodyForce(RigidBody* body, const std::string& name = "SingleBodyForce")
        : Force(name), body_(body) {}

protected:
    RigidBody* body_;
};

/**
 * Force acting between two bodies
 */
class TwoBodyForce : public Force {
public:
    TwoBodyForce(RigidBody* body1, RigidBody* body2,
                 const std::string& name = "TwoBodyForce")
        : Force(name), body1_(body1), body2_(body2) {}

protected:
    RigidBody* body1_;
    RigidBody* body2_;
};

} // namespace mb
