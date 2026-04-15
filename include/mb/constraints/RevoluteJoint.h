#pragma once
#include "Constraint.h"
#include "mb/core/RigidBody.h"

namespace mb {

/**
 * Revolute joint - removes 5 DOF (3 translational + 2 rotational).
 * Allows rotation about a single axis.
 */
class RevoluteJoint : public Constraint {
public:
    RevoluteJoint(RigidBody* body1, RigidBody* body2,
                  const Vec3& localPoint1, const Vec3& localPoint2,
                  const Vec3& localAxis1, const Vec3& localAxis2,
                  const std::string& name = "RevoluteJoint");

    int numEquations() const override { return 5; }
    std::vector<int> getBodyIds() const override;
    ConstraintViolation computeViolation() const override;
    JacobianResult computeJacobian() const override;

    Vec3 getAxis1World() const;
    Vec3 getAxis2World() const;

protected:
    std::vector<double> computeConvectiveTerm() const override;

private:
    RigidBody* body1_;
    RigidBody* body2_;
    Vec3 localPoint1_, localPoint2_;
    Vec3 localAxis1_, localAxis2_;

    // Get two vectors perpendicular to the axis
    std::pair<Vec3, Vec3> getPerpAxes(const Vec3& worldAxis) const;
};

} // namespace mb
