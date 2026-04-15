#include "mb/constraints/RevoluteJoint.h"
#include "mb/math/Mat3.h"
#include <cmath>

namespace mb {

RevoluteJoint::RevoluteJoint(RigidBody* body1, RigidBody* body2,
                             const Vec3& localPoint1, const Vec3& localPoint2,
                             const Vec3& localAxis1, const Vec3& localAxis2,
                             const std::string& name)
    : Constraint(name), body1_(body1), body2_(body2),
      localPoint1_(localPoint1), localPoint2_(localPoint2),
      localAxis1_(localAxis1.normalize()), localAxis2_(localAxis2.normalize()) {}

std::vector<int> RevoluteJoint::getBodyIds() const {
    return {body1_->id, body2_->id};
}

Vec3 RevoluteJoint::getAxis1World() const {
    return body1_->bodyToWorldDirection(localAxis1_);
}

Vec3 RevoluteJoint::getAxis2World() const {
    return body2_->bodyToWorldDirection(localAxis2_);
}

std::pair<Vec3, Vec3> RevoluteJoint::getPerpAxes(const Vec3& worldAxis) const {
    // Find two perpendicular axes
    Vec3 a = worldAxis.normalize();
    Vec3 ref = (std::abs(a.x) < 0.9) ? Vec3::unitX() : Vec3::unitY();
    Vec3 e = a.cross(ref).normalize();
    Vec3 f = a.cross(e).normalize();
    return {e, f};
}

ConstraintViolation RevoluteJoint::computeViolation() const {
    // First 3: position (spherical joint part)
    Vec3 p1 = body1_->bodyToWorld(localPoint1_);
    Vec3 p2 = body2_->bodyToWorld(localPoint2_);
    Vec3 posErr = p1.sub(p2);

    // Last 2: axis alignment (perpendicularity conditions)
    Vec3 axis1W = getAxis1World();
    Vec3 axis2W = getAxis2World();
    auto [e, f] = getPerpAxes(axis2W);

    double dot_e = axis1W.dot(e);
    double dot_f = axis1W.dot(f);

    return {{posErr.x, posErr.y, posErr.z, dot_e, dot_f}, 
            {0, 0, 0, 0, 0}};
}

JacobianResult RevoluteJoint::computeJacobian() const {
    Mat3 R1 = body1_->orientation.toRotationMatrix();
    Mat3 R2 = body2_->orientation.toRotationMatrix();
    Vec3 r1s1 = R1.multiplyVec3(localPoint1_);
    Vec3 r2s2 = R2.multiplyVec3(localPoint2_);

    Mat3 skew1 = r1s1.skewSymmetric();
    Mat3 skew2 = r2s2.skewSymmetric();

    Vec3 axis1W = getAxis1World();
    Vec3 axis2W = getAxis2World();
    auto [e, f] = getPerpAxes(axis2W);

    MatrixN J1(5, 6), J2(5, 6);

    // Position part (rows 0-2): same as spherical joint
    for (int i = 0; i < 3; i++) {
        J1.set(i, i, 1.0);
        J2.set(i, i, -1.0);
        for (int j = 0; j < 3; j++) {
            J1.set(i, 3+j, -skew1.get(i, j));
            J2.set(i, 3+j, skew2.get(i, j));
        }
    }

    // Axis alignment part (rows 3-4)
    // d/dω₁ (a1 · e) = -(a1 × e) (since ȧ₁ = ω₁ × a₁)
    Vec3 a1xe = axis1W.cross(e);
    Vec3 a1xf = axis1W.cross(f);
    Vec3 a2xe = axis2W.cross(e);
    Vec3 a2xf = axis2W.cross(f);

    for (int j = 0; j < 3; j++) {
        // Row 3: d(a1·e)/dω₁, d(a1·e)/dω₂
        J1.set(3, 3+j, -a1xe[j]);  // -(a1 × e)
        J2.set(3, 3+j, 0);         // e depends on body2, but partial is more complex
        // Row 4: d(a1·f)/dω₁, d(a1·f)/dω₂
        J1.set(4, 3+j, -a1xf[j]);
        J2.set(4, 3+j, 0);
    }

    // Body 2 rotational Jacobian for axis constraints
    // ∂(a1·e)/∂ω₂ = a1·(ω₂×e) → a1 · [e]× has sign issues; use cross product identity
    Vec3 exA1 = e.cross(axis1W);
    Vec3 fxA1 = f.cross(axis1W);
    for (int j = 0; j < 3; j++) {
        J2.set(3, 3+j, exA1[j]);
        J2.set(4, 3+j, fxA1[j]);
    }

    return {J1, J2};
}

std::vector<double> RevoluteJoint::computeConvectiveTerm() const {
    // Analytical convective term
    Mat3 R1 = body1_->orientation.toRotationMatrix();
    Mat3 R2 = body2_->orientation.toRotationMatrix();
    Vec3 r1s1 = R1.multiplyVec3(localPoint1_);
    Vec3 r2s2 = R2.multiplyVec3(localPoint2_);
    Vec3 w1 = body1_->angularVelocity;
    Vec3 w2 = body2_->angularVelocity;

    // Position convective: ω × (ω × Rs)
    Vec3 conv1 = w1.cross(w1.cross(r1s1));
    Vec3 conv2 = w2.cross(w2.cross(r2s2));
    Vec3 posConv = conv1.sub(conv2);

    // Axis alignment convective terms
    Vec3 axis1W = getAxis1World();
    Vec3 axis2W = getAxis2World();
    auto [e, f] = getPerpAxes(axis2W);

    Vec3 a1dot = w1.cross(axis1W);
    Vec3 edot = w2.cross(e);
    Vec3 fdot = w2.cross(f);

    double convE = a1dot.dot(e) + axis1W.dot(edot);
    double convF = a1dot.dot(f) + axis1W.dot(fdot);

    return {posConv.x, posConv.y, posConv.z, convE, convF};
}

} // namespace mb
