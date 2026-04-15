#include "mb/constraints/DrivingConstraint.h"
#include <cmath>

namespace mb {

DrivingConstraint::DrivingConstraint(RigidBody* body1, RigidBody* body2,
                                     const Vec3& localAxis1, const Vec3& localAxis2,
                                     double angularVelocity, double initialAngle,
                                     const std::string& name)
    : Constraint(name), body1_(body1), body2_(body2),
      localAxis1_(localAxis1.normalize()), localAxis2_(localAxis2.normalize()),
      omega_(angularVelocity), theta0_(initialAngle) {}

std::vector<int> DrivingConstraint::getBodyIds() const {
    return {body1_->id, body2_->id};
}

double DrivingConstraint::measureAngle() const {
    Vec3 axis1W = body1_->bodyToWorldDirection(localAxis1_);
    Vec3 axis2W = body2_->bodyToWorldDirection(localAxis2_);

    // Find reference directions perpendicular to axes
    Vec3 ref = (std::abs(axis1W.x) < 0.9) ? Vec3::unitX() : Vec3::unitY();
    Vec3 e1 = axis1W.cross(ref).normalize();
    Vec3 e2 = axis2W.cross(ref).normalize();

    double cosAngle = e1.dot(e2);
    double sinAngle = axis1W.dot(e1.cross(e2));

    return std::atan2(sinAngle, cosAngle);
}

ConstraintViolation DrivingConstraint::computeViolation() const {
    double theta = measureAngle();
    double target = omega_ * time + theta0_;

    // Wrap to [-π, π]
    double err = theta - target;
    while (err > M_PI) err -= 2 * M_PI;
    while (err < -M_PI) err += 2 * M_PI;

    return {{err}, {0}};
}

JacobianResult DrivingConstraint::computeJacobian() const {
    Vec3 axis1W = body1_->bodyToWorldDirection(localAxis1_);

    MatrixN J1(1, 6), J2(1, 6);

    // The angle θ depends on the relative orientation.
    // ∂θ/∂ω₁ = axis1W (rotation about axis changes θ)
    // ∂θ/∂ω₂ = -axis2W
    for (int j = 0; j < 3; j++) {
        J1.set(0, 3+j, axis1W[j]);
        J2.set(0, 3+j, -axis1W[j]);
    }

    return {J1, J2};
}

std::vector<double> DrivingConstraint::computeConvectiveTerm() const {
    // For rheonomic constraint: γ includes C_tt term
    // C_t = -ω (prescribed angular velocity)
    // Convective = 0 for linear-in-time driving
    // But Baumgarte needs C_t contribution
    // This is handled in getGamma via the velocity violation
    return {0.0};
}

} // namespace mb
