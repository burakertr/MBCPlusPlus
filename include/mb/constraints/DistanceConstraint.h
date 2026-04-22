#pragma once
#include "Constraint.h"
#include "mb/core/RigidBody.h"

namespace mb {

/**
 * Distance constraint - maintains a fixed distance between two points.
 * C = |p1 - p2| - L = 0
 */
class DistanceConstraint : public Constraint {
public:
    DistanceConstraint(RigidBody* body1, RigidBody* body2,
                       const Vec3& localPoint1, const Vec3& localPoint2,
                       double distance,
                       const std::string& name = "DistanceConstraint");

    int numEquations() const override { return 1; }
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
    double distance_;
    bool useSquaredForm_ = false;
};

} // namespace mb
