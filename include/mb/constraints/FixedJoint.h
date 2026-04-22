#pragma once
#include "Constraint.h"
#include "mb/core/RigidBody.h"

namespace mb {

/**
 * Fixed joint - removes all 6 DOF (3 translational + 3 rotational).
 */
class FixedJoint : public Constraint {
public:
    FixedJoint(RigidBody* body1, RigidBody* body2,
               const Vec3& localPoint1, const Vec3& localPoint2,
               const std::string& name = "FixedJoint");

    int numEquations() const override { return 6; }
    std::vector<int> getBodyIds() const override;
    ConstraintViolation computeViolation() const override;
    JacobianResult computeJacobian() const override;
    std::vector<double> computeVelocityViolation() const override;

protected:
    std::vector<double> computeConvectiveTerm() const override;

private:
    RigidBody* body1_;
    RigidBody* body2_;
    Vec3 localPoint1_, localPoint2_;
    Quaternion refOrientation_; // Initial relative orientation
};

} // namespace mb
