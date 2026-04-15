#pragma once
#include "Force.h"
#include "mb/core/RigidBody.h"
#include <vector>

namespace mb {

/**
 * Uniform gravity force applied to a set of bodies
 */
class Gravity : public Force {
public:
    Gravity(const Vec3& g = Vec3(0, 0, -9.81));

    void setGravity(const Vec3& g) { gravity_ = g; }
    Vec3 getGravity() const { return gravity_; }

    void addBody(RigidBody* body);
    void removeBody(RigidBody* body);

    void apply(double t) override;

private:
    Vec3 gravity_;
    std::vector<RigidBody*> bodies_;
};

} // namespace mb
