#include "mb/constraints/SphericalJoint.h"
#include "mb/math/Mat3.h"
#include <cmath>

namespace mb {

SphericalJoint::SphericalJoint(RigidBody* body1, RigidBody* body2,
                               const Vec3& localPoint1, const Vec3& localPoint2,
                               const std::string& name)
    : Constraint(name), body1_(body1), body2_(body2),
      localPoint1_(localPoint1), localPoint2_(localPoint2) {}

std::vector<int> SphericalJoint::getBodyIds() const {
    return {body1_->id, body2_->id};
}

ConstraintViolation SphericalJoint::computeViolation() const {
    Vec3 p1 = body1_->bodyToWorld(localPoint1_);
    Vec3 p2 = body2_->bodyToWorld(localPoint2_);
    Vec3 diff = p1.sub(p2);
    return {{diff.x, diff.y, diff.z}, {0, 0, 0}};
}

JacobianResult SphericalJoint::computeJacobian() const {
    Mat3 R1 = body1_->orientation.toRotationMatrix();
    Mat3 R2 = body2_->orientation.toRotationMatrix();
    Vec3 r1s1 = R1.multiplyVec3(localPoint1_);
    Vec3 r2s2 = R2.multiplyVec3(localPoint2_);

    // Skew symmetric matrices
    Mat3 skew1 = r1s1.skewSymmetric();
    Mat3 skew2 = r2s2.skewSymmetric();

    // J1 = [I₃  -[R1*s1]×]
    // J2 = [-I₃  [R2*s2]×]
    MatrixN J1(3, 6), J2(3, 6);
    for (int i = 0; i < 3; i++) {
        J1.set(i, i, 1.0);        // I₃
        J2.set(i, i, -1.0);       // -I₃
        for (int j = 0; j < 3; j++) {
            J1.set(i, 3+j, -skew1.get(i, j));  // -[R1*s1]×
            J2.set(i, 3+j, skew2.get(i, j));    // [R2*s2]×
        }
    }

    return {J1, J2};
}

std::vector<double> SphericalJoint::computeVelocityViolation() const {
    // Ċ = v1 + ω1 × (R1*s1) - v2 - ω2 × (R2*s2)
    Mat3 R1 = body1_->orientation.toRotationMatrix();
    Mat3 R2 = body2_->orientation.toRotationMatrix();
    Vec3 r1s1 = R1.multiplyVec3(localPoint1_);
    Vec3 r2s2 = R2.multiplyVec3(localPoint2_);

    Vec3 cdot = body1_->velocity + body1_->angularVelocity.cross(r1s1)
              - body2_->velocity - body2_->angularVelocity.cross(r2s2);
    return {cdot.x, cdot.y, cdot.z};
}

std::vector<double> SphericalJoint::computeConvectiveTerm() const {
    // Analytical convective term: ω × (ω × R*s) for each body
    Mat3 R1 = body1_->orientation.toRotationMatrix();
    Mat3 R2 = body2_->orientation.toRotationMatrix();
    Vec3 r1s1 = R1.multiplyVec3(localPoint1_);
    Vec3 r2s2 = R2.multiplyVec3(localPoint2_);

    Vec3 w1 = body1_->angularVelocity;
    Vec3 w2 = body2_->angularVelocity;

    Vec3 conv1 = w1.cross(w1.cross(r1s1));
    Vec3 conv2 = w2.cross(w2.cross(r2s2));
    Vec3 conv = conv1.sub(conv2);

    return {conv.x, conv.y, conv.z};
}

} // namespace mb
