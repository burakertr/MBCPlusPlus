#pragma once
#include "mb/contact/ContactTypes.h"
#include "mb/core/Body.h"
#include "mb/core/RigidBody.h"
#include <vector>
#include <functional>

namespace mb {

/**
 * Collision detection between rigid bodies.
 * Supports sphere, box, cylinder, cone vs ground plane,
 * sphere-sphere, sphere-box, and sphere-cylinder.
 * Falls back to bounding-sphere for unsupported pairs.
 */
class CollisionDetector {
public:
    CollisionDetector();

    /// Detect all contacts between a set of bodies
    std::vector<ContactPair> detectCollisions(
        const std::vector<std::shared_ptr<Body>>& bodies) const;

    /// Detect contact between two specific bodies
    std::vector<ContactPair> detectPair(Body* a, Body* b) const;

    void setContactThreshold(double threshold) { threshold_ = threshold; }
    double getContactThreshold() const { return threshold_; }

private:
    double threshold_ = 0.01;

    // Shape-specific collision routines
    std::vector<ContactPair> sphereVsPlane(RigidBody* sphere, RigidBody* plane) const;
    std::vector<ContactPair> sphereVsSphere(RigidBody* a, RigidBody* b) const;
    std::vector<ContactPair> boxVsPlane(RigidBody* box, RigidBody* plane) const;
    std::vector<ContactPair> sphereVsBox(RigidBody* sphere, RigidBody* box) const;
    std::vector<ContactPair> cylinderVsPlane(RigidBody* cyl, RigidBody* plane) const;
    std::vector<ContactPair> sphereVsCylinder(RigidBody* sphere, RigidBody* cyl) const;
    std::vector<ContactPair> boundingSphereFallback(RigidBody* a, RigidBody* b) const;

    // Helper: approximate bounding sphere radius
    double getBoundingRadius(RigidBody* body) const;
};

} // namespace mb
