#pragma once
#include "Constraint.h"
#include "mb/core/RigidBody.h"

namespace mb {

/**
 * Spherical joint - removes 3 translational DOF.
 * C = r1 + R1*s1 - r2 - R2*s2 = 0
 */
class SphericalJoint : public Constraint {
public:
    SphericalJoint(RigidBody* body1, RigidBody* body2,
                   const Vec3& localPoint1, const Vec3& localPoint2,
                   const std::string& name = "SphericalJoint");

    int numEquations() const override { return 3; }
    std::vector<int> getBodyIds() const override;
    ConstraintViolation computeViolation() const override;
    JacobianResult computeJacobian() const override;
    std::vector<double> computeVelocityViolation() const override;

protected:
    std::vector<double> computeConvectiveTerm() const override;

private:
    RigidBody* body1_;
    RigidBody* body2_;
    Vec3 localPoint1_;
    Vec3 localPoint2_;
};

} // namespace mb
