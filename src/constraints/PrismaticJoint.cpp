#include "mb/constraints/PrismaticJoint.h"
#include "mb/math/Mat3.h"
#include <cmath>

namespace mb {

PrismaticJoint::PrismaticJoint(RigidBody* body1, RigidBody* body2,
                               const Vec3& localPoint1, const Vec3& localPoint2,
                               const Vec3& localAxis1, const Vec3& localAxis2,
                               const std::string& name)
    : Constraint(name), body1_(body1), body2_(body2),
      localPoint1_(localPoint1), localPoint2_(localPoint2),
      localAxis1_(localAxis1.normalize()), localAxis2_(localAxis2.normalize()) {
    // Store reference relative orientation
    refOrientation_ = body2_->orientation.inverse().multiply(body1_->orientation);
}

std::vector<int> PrismaticJoint::getBodyIds() const {
    return {body1_->id, body2_->id};
}

ConstraintViolation PrismaticJoint::computeViolation() const {
    Vec3 p1 = body1_->bodyToWorld(localPoint1_);
    Vec3 p2 = body2_->bodyToWorld(localPoint2_);
    Vec3 d = p1.sub(p2);
    Vec3 axisW = body1_->bodyToWorldDirection(localAxis1_);

    // 2 perpendicular position constraints
    Vec3 ref = (std::abs(axisW.x) < 0.9) ? Vec3::unitX() : Vec3::unitY();
    Vec3 e = axisW.cross(ref).normalize();
    Vec3 f = axisW.cross(e).normalize();

    double c1 = d.dot(e);
    double c2 = d.dot(f);

    // 3 orientation constraints (using cross-product error)
    Quaternion qRel = body2_->orientation.inverse().multiply(body1_->orientation);
    Quaternion qErr = qRel.multiply(refOrientation_.inverse());
    // Small-angle: error ≈ 2 * vector part
    double c3 = 2.0 * qErr.x;
    double c4 = 2.0 * qErr.y;
    double c5 = 2.0 * qErr.z;

    return {{c1, c2, c3, c4, c5}, {0, 0, 0, 0, 0}};
}

JacobianResult PrismaticJoint::computeJacobian() const {
    Mat3 R1 = body1_->orientation.toRotationMatrix();
    Vec3 r1s1 = R1.multiplyVec3(localPoint1_);
    Mat3 R2 = body2_->orientation.toRotationMatrix();
    Vec3 r2s2 = R2.multiplyVec3(localPoint2_);
    Mat3 skew1 = r1s1.skewSymmetric();
    Mat3 skew2 = r2s2.skewSymmetric();

    Vec3 axisW = body1_->bodyToWorldDirection(localAxis1_);
    Vec3 ref = (std::abs(axisW.x) < 0.9) ? Vec3::unitX() : Vec3::unitY();
    Vec3 e = axisW.cross(ref).normalize();
    Vec3 f = axisW.cross(e).normalize();

    MatrixN J1(5, 6), J2(5, 6);

    // Position constraints (rows 0-1): d·e and d·f
    Vec3 p1 = body1_->bodyToWorld(localPoint1_);
    Vec3 p2 = body2_->bodyToWorld(localPoint2_);
    Vec3 d = p1.sub(p2);

    for (int j = 0; j < 3; j++) {
        J1.set(0, j, e[j]);
        J1.set(1, j, f[j]);
        J2.set(0, j, -e[j]);
        J2.set(1, j, -f[j]);
    }

    // Angular part of position constraints
    Vec3 skew1_e_cross = r1s1.cross(e).negate();
    Vec3 skew1_f_cross = r1s1.cross(f).negate();
    Vec3 skew2_e_cross = r2s2.cross(e);
    Vec3 skew2_f_cross = r2s2.cross(f);

    for (int j = 0; j < 3; j++) {
        J1.set(0, 3+j, skew1_e_cross[j]);
        J1.set(1, 3+j, skew1_f_cross[j]);
        J2.set(0, 3+j, skew2_e_cross[j]);
        J2.set(1, 3+j, skew2_f_cross[j]);
    }

    // Orientation constraints (rows 2-4): relative orientation via identity Jacobian
    for (int i = 0; i < 3; i++) {
        J1.set(2+i, 3+i, 1.0);
        J2.set(2+i, 3+i, -1.0);
    }

    return {J1, J2};
}

} // namespace mb
