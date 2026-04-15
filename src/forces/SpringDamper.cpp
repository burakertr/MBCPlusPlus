#include "mb/forces/SpringDamper.h"
#include <cmath>

namespace mb {

SpringDamper::SpringDamper(RigidBody* body1, RigidBody* body2,
                           const Vec3& localPoint1, const Vec3& localPoint2,
                           double stiffness, double damping, double restLength,
                           const std::string& name)
    : TwoBodyForce(body1, body2, name),
      localPoint1_(localPoint1), localPoint2_(localPoint2),
      stiffness_(stiffness), damping_(damping), restLength_(restLength) {}

SpringDamper SpringDamper::criticallyDamped(
    RigidBody* body1, RigidBody* body2,
    const Vec3& localPoint1, const Vec3& localPoint2,
    double stiffness, double restLength,
    const std::string& name
) {
    // Critical damping: c = 2 * sqrt(k * m_eff)
    double m1 = body1->mass;
    double m2 = body2->mass;
    double mEff = (m1 > 0 && m2 > 0) ? (m1 * m2) / (m1 + m2) : std::max(m1, m2);
    double damping = 2.0 * std::sqrt(stiffness * mEff);
    return SpringDamper(body1, body2, localPoint1, localPoint2,
                        stiffness, damping, restLength, name);
}

void SpringDamper::apply(double /*t*/) {
    Vec3 p1 = body1_->bodyToWorld(localPoint1_);
    Vec3 p2 = body2_->bodyToWorld(localPoint2_);
    Vec3 d = p2.sub(p1);
    double dist = d.length();

    if (dist < 1e-12) return;

    Vec3 n = d.scale(1.0 / dist);

    // Spring force: F = -k * (dist - L0) * n
    double stretch = dist - restLength_;
    Vec3 springForce = n.scale(-stiffness_ * stretch);

    // Damper force: F = -c * (v_rel · n) * n
    Vec3 v1 = body1_->getPointVelocity(p1);
    Vec3 v2 = body2_->getPointVelocity(p2);
    Vec3 vRel = v2.sub(v1);
    double vn = vRel.dot(n);
    Vec3 damperForce = n.scale(-damping_ * vn);

    Vec3 totalForce = springForce.add(damperForce);

    // Apply equal and opposite forces
    body1_->applyForceAtPoint(totalForce.negate(), p1);
    body2_->applyForceAtPoint(totalForce, p2);
}

} // namespace mb
