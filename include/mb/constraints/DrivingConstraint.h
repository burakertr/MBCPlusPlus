#pragma once
#include "Constraint.h"
#include "mb/core/RigidBody.h"

namespace mb {

/**
 * Driving constraint - prescribes relative rotation angle.
 * C = θ_rel - (ω*t + θ₀) = 0  (rheonomic)
 */
class DrivingConstraint : public Constraint {
public:
    DrivingConstraint(RigidBody* body1, RigidBody* body2,
                      const Vec3& localAxis1, const Vec3& localAxis2,
                      double angularVelocity, double initialAngle = 0.0,
                      const std::string& name = "DrivingConstraint");

    int numEquations() const override { return 1; }
    std::vector<int> getBodyIds() const override;
    ConstraintViolation computeViolation() const override;
    JacobianResult computeJacobian() const override;

protected:
    std::vector<double> computeConvectiveTerm() const override;

private:
    RigidBody* body1_;
    RigidBody* body2_;
    Vec3 localAxis1_, localAxis2_;
    double omega_;
    double theta0_;

    double measureAngle() const;
};

} // namespace mb
