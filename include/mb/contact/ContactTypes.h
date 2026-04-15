#pragma once
#include "mb/math/Vec3.h"
#include <vector>
#include <memory>
#include <cstdint>

namespace mb {

class Body;

/**
 * Material properties for contact force computation
 */
struct ContactMaterial {
    double restitution = 0.5;     // Coefficient of restitution [0,1]
    double friction = 0.3;        // Coulomb friction coefficient
    double stiffness = 1e5;       // Contact stiffness (Hertz)
    double damping = 1e3;         // Contact damping
    double rollingFriction = 0.0; // Rolling friction coefficient
};

/**
 * Geometric collision pair
 */
struct ContactPair {
    Body* bodyA = nullptr;
    Body* bodyB = nullptr;
    Vec3 pointA;         // Contact point on body A (world)
    Vec3 pointB;         // Contact point on body B (world)
    Vec3 normal;         // Contact normal (A→B)
    double penetration = 0.0; // Penetration depth (>0 means overlap)
};

/**
 * An active contact with computed forces
 */
struct ActiveContact {
    ContactPair pair;
    Vec3 normalForce;
    Vec3 frictionForce;
    Vec3 totalForce;
    double normalImpulse = 0.0;
    double tangentImpulse = 0.0;
    bool isActive = true;
    int lifetime = 0;       // Frames this contact has been alive
};

/**
 * Contact detection/resolution configuration
 */
struct ContactConfig {
    double contactThreshold = 0.01;   // Distance below which contact is detected
    double separationThreshold = 0.02;
    bool useFriction = true;
    bool useRestitution = true;
    int maxContactsPerPair = 4;
    double baumgarteAlpha = 0.2;      // Position correction fraction
    double slopDistance = 0.005;       // Penetration slop
};

/**
 * Default material properties
 */
inline ContactMaterial defaultMaterial() {
    return ContactMaterial{};
}

} // namespace mb
