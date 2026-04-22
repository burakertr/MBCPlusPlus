#include "mb/constraints/FixedJoint.h"
#include "mb/math/Mat3.h"
#include <cmath>

namespace mb {

FixedJoint::FixedJoint(RigidBody* body1, RigidBody* body2,
                       const Vec3& localPoint1, const Vec3& localPoint2,
                       const std::string& name)
    : Constraint(name), body1_(body1), body2_(body2),
      localPoint1_(localPoint1), localPoint2_(localPoint2) {
    // Store initial relative orientation
    refOrientation_ = body2_->orientation.inverse().multiply(body1_->orientation);
}

std::vector<int> FixedJoint::getBodyIds() const {
    return {body1_->id, body2_->id};
}

ConstraintViolation FixedJoint::computeViolation() const {
    // Position: same as spherical
    Vec3 p1 = body1_->bodyToWorld(localPoint1_);
    Vec3 p2 = body2_->bodyToWorld(localPoint2_);
    Vec3 posErr = p1.sub(p2);

    // Orientation: cross-product error
    Quaternion qRel = body2_->orientation.inverse().multiply(body1_->orientation);
    Quaternion qErr = qRel.multiply(refOrientation_.inverse());
    // Ensure w > 0 for consistent sign
    if (qErr.w < 0) qErr = Quaternion(-qErr.w, -qErr.x, -qErr.y, -qErr.z);

    return {{posErr.x, posErr.y, posErr.z, 2.0*qErr.x, 2.0*qErr.y, 2.0*qErr.z},
            {0, 0, 0, 0, 0, 0}};
}

JacobianResult FixedJoint::computeJacobian() const {
    Mat3 R1 = body1_->orientation.toRotationMatrix();
    Mat3 R2 = body2_->orientation.toRotationMatrix();
    Vec3 r1s1 = R1.multiplyVec3(localPoint1_);
    Vec3 r2s2 = R2.multiplyVec3(localPoint2_);
    Mat3 skew1 = r1s1.skewSymmetric();
    Mat3 skew2 = r2s2.skewSymmetric();

    MatrixN J1(6, 6), J2(6, 6);

    // Position part (rows 0-2)
    for (int i = 0; i < 3; i++) {
        J1.set(i, i, 1.0);
        J2.set(i, i, -1.0);
        for (int j = 0; j < 3; j++) {
            J1.set(i, 3+j, -skew1.get(i, j));
            J2.set(i, 3+j, skew2.get(i, j));
        }
    }

    // Orientation part (rows 3-5): ω₁ - ω₂ = 0
    for (int i = 0; i < 3; i++) {
        J1.set(3+i, 3+i, 1.0);
        J2.set(3+i, 3+i, -1.0);
    }

    return {J1, J2};
}

std::vector<double> FixedJoint::computeVelocityViolation() const {
    // Position part: v1 + ω1×(R1*s1) - v2 - ω2×(R2*s2)
    Mat3 R1 = body1_->orientation.toRotationMatrix();
    Mat3 R2 = body2_->orientation.toRotationMatrix();
    Vec3 r1s1 = R1.multiplyVec3(localPoint1_);
    Vec3 r2s2 = R2.multiplyVec3(localPoint2_);
    Vec3 vRel = body1_->velocity + body1_->angularVelocity.cross(r1s1)
              - body2_->velocity - body2_->angularVelocity.cross(r2s2);

    // Orientation part: d/dt(2*qErr.vec) ≈ ω1 - ω2
    Vec3 wRel = body1_->angularVelocity.sub(body2_->angularVelocity);

    return {vRel.x, vRel.y, vRel.z, wRel.x, wRel.y, wRel.z};
}

std::vector<double> FixedJoint::computeConvectiveTerm() const {
    // Position convective: ω×(ω×Rs)
    Mat3 R1 = body1_->orientation.toRotationMatrix();
    Mat3 R2 = body2_->orientation.toRotationMatrix();
    Vec3 r1s1 = R1.multiplyVec3(localPoint1_);
    Vec3 r2s2 = R2.multiplyVec3(localPoint2_);
    Vec3 w1 = body1_->angularVelocity;
    Vec3 w2 = body2_->angularVelocity;
    Vec3 posConv = w1.cross(w1.cross(r1s1)).sub(w2.cross(w2.cross(r2s2)));
    // Orientation convective: d/dt(ω1-ω2) = α1-α2 (zero for free rotation — handled by Jacobian)
    return {posConv.x, posConv.y, posConv.z, 0.0, 0.0, 0.0};
}

} // namespace mb
