#include "mb/contact/ContactForceModel.h"
#include "mb/core/Body.h"
#include <cmath>
#include <algorithm>

namespace mb {

ContactForceModel::ContactForceModel() {}

std::vector<ActiveContact> ContactForceModel::computeForces(
    const std::vector<ContactPair>& pairs, double dt) const {

    std::vector<ActiveContact> active;
    active.reserve(pairs.size());
    for (auto& p : pairs) {
        auto ac = computeSingleForce(p, dt);
        if (ac.isActive)
            active.push_back(ac);
    }
    return active;
}

ActiveContact ContactForceModel::computeSingleForce(
    const ContactPair& pair, double dt) const {

    ActiveContact ac;
    ac.pair = pair;

    if (pair.penetration <= 0.0) {
        ac.isActive = false;
        return ac;
    }

    // Relative velocity at contact
    Vec3 relVel = Vec3::zero();
    if (pair.bodyA) {
        relVel = relVel + pair.bodyA->velocity;
        Vec3 rA = pair.pointA - pair.bodyA->position;
        relVel = relVel + pair.bodyA->angularVelocity.cross(rA);
    }
    if (pair.bodyB) {
        relVel = relVel - pair.bodyB->velocity;
        Vec3 rB = pair.pointB - pair.bodyB->position;
        relVel = relVel - pair.bodyB->angularVelocity.cross(rB);
    }

    double vn = relVel.dot(pair.normal); // Normal approach velocity
    Vec3 vt = relVel - pair.normal * vn; // Tangential velocity

    // Normal force: Hertz + Flores damping
    double fn = computeNormalForce(pair.penetration, vn, dt);
    ac.normalForce = pair.normal * fn;
    ac.normalImpulse = fn * dt;

    // Friction force
    ac.frictionForce = computeFrictionForce(fn, vt);
    ac.tangentImpulse = ac.frictionForce.length() * dt;

    ac.totalForce = ac.normalForce + ac.frictionForce;
    ac.isActive = true;
    return ac;
}

double ContactForceModel::computeNormalForce(
    double penetration, double vn, double /*dt*/) const {

    // Hertz: f_n = k * δ^(3/2)
    double hertz = material_.stiffness * std::pow(penetration, 1.5);

    // Flores damping: proportional to penetration to avoid discontinuity
    // f_d = d * δ * δ̇  (only when approaching, vn < 0)
    double damping = 0.0;
    if (vn < 0.0) {
        damping = material_.damping * penetration * (-vn);
    }

    // Restitution scaling
    double e = material_.restitution;
    double scale = 1.0 + 8.0 * (1.0 - e) / (5.0 * e + 1e-10);
    damping *= scale;

    double fn = hertz + damping;
    return std::max(0.0, fn); // Normal force must be repulsive
}

Vec3 ContactForceModel::computeFrictionForce(
    double normalForce, const Vec3& vt) const {

    double vtMag = vt.length();
    if (vtMag < 1e-12) return Vec3::zero();

    // Regularised Coulomb: smooth transition near zero velocity
    double mu = material_.friction;
    double regVel = 0.01; // Regularisation velocity
    double frictionMag = mu * normalForce * std::tanh(vtMag / regVel);

    Vec3 dir = vt * (-1.0 / vtMag); // Oppose sliding direction
    return dir * frictionMag;
}

} // namespace mb
