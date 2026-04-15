#include "mb/contact/ContactManager.h"
#include <cmath>
#include <algorithm>

namespace mb {

ContactManager::ContactManager() {}

void ContactManager::setMaterial(const ContactMaterial& mat) {
    forceModel_.setMaterial(mat);
}

const ContactMaterial& ContactManager::getMaterial() const {
    return forceModel_.getMaterial();
}

void ContactManager::processContacts(
    const std::vector<std::shared_ptr<Body>>& bodies, double dt) {

    // 1. Detect collisions
    detector_.setContactThreshold(config_.contactThreshold);
    auto pairs = detector_.detectCollisions(bodies);

    // 2. Update lifetime tracking
    updateContactLifetimes(pairs);

    // 3. Compute contact forces
    activeContacts_ = forceModel_.computeForces(pairs, dt);

    // 4. Apply forces to bodies
    applyContactForces();
}

void ContactManager::resolveImpulses(
    const std::vector<std::shared_ptr<Body>>& bodies, double dt) {

    // Sequential impulse position correction
    for (auto& ac : activeContacts_) {
        if (!ac.isActive) continue;
        double pen = ac.pair.penetration - config_.slopDistance;
        if (pen <= 0.0) continue;

        Body* a = ac.pair.bodyA;
        Body* b = ac.pair.bodyB;

        double invMassA = (a && a->isDynamic()) ? 1.0 / a->getMass() : 0.0;
        double invMassB = (b && b->isDynamic()) ? 1.0 / b->getMass() : 0.0;
        double invMassSum = invMassA + invMassB;
        if (invMassSum < 1e-15) continue;

        Vec3 correction = ac.pair.normal * (config_.baumgarteAlpha * pen / invMassSum);

        if (a && a->isDynamic())
            a->position = a->position - correction * invMassA;
        if (b && b->isDynamic())
            b->position = b->position + correction * invMassB;

        // Velocity correction for restitution
        double vn = 0.0;
        if (a) vn += a->velocity.dot(ac.pair.normal);
        if (b) vn -= b->velocity.dot(ac.pair.normal);

        if (vn < 0.0) {
            double e = forceModel_.getMaterial().restitution;
            double jn = -(1.0 + e) * vn / invMassSum;
            Vec3 impulse = ac.pair.normal * jn;

            if (a && a->isDynamic())
                a->velocity = a->velocity + impulse * invMassA;
            if (b && b->isDynamic())
                b->velocity = b->velocity - impulse * invMassB;
        }
    }
}

void ContactManager::applyContactForces() {
    for (auto& ac : activeContacts_) {
        if (!ac.isActive) continue;
        Body* a = ac.pair.bodyA;
        Body* b = ac.pair.bodyB;

        if (a && a->isDynamic()) {
            a->applyForce(ac.totalForce);
            Vec3 rA = ac.pair.pointA - a->position;
            a->applyTorque(rA.cross(ac.totalForce));
        }
        if (b && b->isDynamic()) {
            Vec3 neg = ac.totalForce * -1.0;
            b->applyForce(neg);
            Vec3 rB = ac.pair.pointB - b->position;
            b->applyTorque(rB.cross(neg));
        }
    }
}

void ContactManager::updateContactLifetimes(
    const std::vector<ContactPair>& /*newPairs*/) {
    // Simple approach: increment lifetime for existing contacts
    for (auto& ac : activeContacts_)
        ac.lifetime++;
}

double ContactManager::getTotalContactForce() const {
    double total = 0.0;
    for (auto& ac : activeContacts_) {
        if (ac.isActive)
            total += ac.totalForce.length();
    }
    return total;
}

void ContactManager::clear() {
    activeContacts_.clear();
}

} // namespace mb
