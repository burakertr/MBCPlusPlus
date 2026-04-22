#pragma once
#include "Constraint.h"
#include "mb/core/RigidBody.h"

namespace mb {

/**
 * Prismatic joint - removes 5 DOF (2 translational + 3 rotational).
 * Allows translation along a single axis.
 */
class PrismaticJoint : public Constraint {
public:
    PrismaticJoint(RigidBody* body1, RigidBody* body2,
                   const Vec3& localPoint1, const Vec3& localPoint2,
                   const Vec3& localAxis1, const Vec3& localAxis2,
                   const std::string& name = "PrismaticJoint");

    int numEquations() const override { return 5; }
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
    Vec3 localAxis1_, localAxis2_;
    Quaternion refOrientation_; // Reference relative orientation
};

} // namespace mb
