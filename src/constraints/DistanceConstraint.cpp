#include "mb/constraints/DistanceConstraint.h"
#include <cmath>

namespace mb {

DistanceConstraint::DistanceConstraint(RigidBody* body1, RigidBody* body2,
                                       const Vec3& localPoint1, const Vec3& localPoint2,
                                       double distance, const std::string& name)
    : Constraint(name), body1_(body1), body2_(body2),
      localPoint1_(localPoint1), localPoint2_(localPoint2), distance_(distance) {}

std::vector<int> DistanceConstraint::getBodyIds() const {
    return {body1_->id, body2_->id};
}

ConstraintViolation DistanceConstraint::computeViolation() const {
    Vec3 p1 = body1_->bodyToWorld(localPoint1_);
    Vec3 p2 = body2_->bodyToWorld(localPoint2_);
    Vec3 d = p1.sub(p2);
    double dist = d.length();
    double C = dist - distance_;
    return {{C}, {0}};
}

JacobianResult DistanceConstraint::computeJacobian() const {
    Vec3 p1 = body1_->bodyToWorld(localPoint1_);
    Vec3 p2 = body2_->bodyToWorld(localPoint2_);
    Vec3 d = p1.sub(p2);
    double dist = d.length();

    Vec3 n = (dist > 1e-12) ? d.scale(1.0 / dist) : Vec3::unitX();

    Mat3 R1 = body1_->orientation.toRotationMatrix();
    Mat3 R2 = body2_->orientation.toRotationMatrix();
    Vec3 r1s1 = R1.multiplyVec3(localPoint1_);
    Vec3 r2s2 = R2.multiplyVec3(localPoint2_);

    MatrixN J1(1, 6), J2(1, 6);

    // J = ∂C/∂q = n^T * [∂p1/∂q - ∂p2/∂q]
    for (int j = 0; j < 3; j++) {
        J1.set(0, j, n[j]);
        J2.set(0, j, -n[j]);
    }

    // Angular part: n^T * (-[R*s]×)
    Vec3 nxRs1 = n.cross(r1s1).negate();
    Vec3 nxRs2 = n.cross(r2s2);
    // Actually: J_rot = n^T * d(Rs)/dω = n^T * (-[Rs]×) = ([Rs]×)^T * n = -(Rs × n)
    // Wait, let me be more careful:
    // dp1/dω1 = -[R1*s1]× so J1_rot = n^T * (-[R1*s1]×) = (R1*s1 × n)
    Vec3 Rs1xn = r1s1.cross(n);
    Vec3 Rs2xn = r2s2.cross(n);

    for (int j = 0; j < 3; j++) {
        J1.set(0, 3+j, -Rs1xn[j]);
        J2.set(0, 3+j, Rs2xn[j]);
    }

    return {J1, J2};
}

std::vector<double> DistanceConstraint::computeConvectiveTerm() const {
    Vec3 p1 = body1_->bodyToWorld(localPoint1_);
    Vec3 p2 = body2_->bodyToWorld(localPoint2_);
    Vec3 d = p1.sub(p2);
    double dist = d.length();
    if (dist < 1e-12) return {0.0};

    Vec3 n = d.scale(1.0 / dist);

    // Velocity of attachment points
    Vec3 v1 = body1_->getPointVelocity(p1);
    Vec3 v2 = body2_->getPointVelocity(p2);
    Vec3 vRel = v1.sub(v2);

    // Convective: (|vRel|² - (n·vRel)²) / dist
    double vn = vRel.dot(n);
    double convective = (vRel.lengthSquared() - vn * vn) / dist;

    return {convective};
}

} // namespace mb
