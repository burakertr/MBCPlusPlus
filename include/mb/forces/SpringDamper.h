#pragma once
#include "Force.h"
#include "mb/core/RigidBody.h"

namespace mb {

/**
 * Linear spring-damper between two bodies.
 * F = -k*(|d|-L₀)*n - c*(vRel·n)*n
 */
class SpringDamper : public TwoBodyForce {
public:
    SpringDamper(RigidBody* body1, RigidBody* body2,
                 const Vec3& localPoint1, const Vec3& localPoint2,
                 double stiffness, double damping, double restLength,
                 const std::string& name = "SpringDamper");

    static SpringDamper criticallyDamped(
        RigidBody* body1, RigidBody* body2,
        const Vec3& localPoint1, const Vec3& localPoint2,
        double stiffness, double restLength,
        const std::string& name = "SpringDamper");

    void apply(double t) override;

    double getStiffness() const { return stiffness_; }
    double getDamping() const { return damping_; }
    double getRestLength() const { return restLength_; }

private:
    Vec3 localPoint1_, localPoint2_;
    double stiffness_;
    double damping_;
    double restLength_;
};

} // namespace mb
