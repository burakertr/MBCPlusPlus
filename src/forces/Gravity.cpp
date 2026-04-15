#include "mb/forces/Gravity.h"
#include <algorithm>

namespace mb {

Gravity::Gravity(const Vec3& g) : Force("Gravity"), gravity_(g) {}

void Gravity::addBody(RigidBody* body) {
    bodies_.push_back(body);
}

void Gravity::removeBody(RigidBody* body) {
    bodies_.erase(
        std::remove(bodies_.begin(), bodies_.end(), body),
        bodies_.end()
    );
}

void Gravity::apply(double /*t*/) {
    for (auto* body : bodies_) {
        if (body->isDynamic()) {
            body->applyForce(gravity_.scale(body->mass));
        }
    }
}

} // namespace mb
