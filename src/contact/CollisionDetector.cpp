#include "mb/contact/CollisionDetector.h"
#include <algorithm>
#include <cmath>

namespace mb {

CollisionDetector::CollisionDetector() {}

std::vector<ContactPair> CollisionDetector::detectCollisions(
    const std::vector<std::shared_ptr<Body>>& bodies) const {
    std::vector<ContactPair> contacts;
    for (size_t i = 0; i < bodies.size(); i++) {
        for (size_t j = i + 1; j < bodies.size(); j++) {
            auto pairs = detectPair(bodies[i].get(), bodies[j].get());
            contacts.insert(contacts.end(), pairs.begin(), pairs.end());
        }
    }
    return contacts;
}

std::vector<ContactPair> CollisionDetector::detectPair(Body* a, Body* b) const {
    auto* ra = dynamic_cast<RigidBody*>(a);
    auto* rb = dynamic_cast<RigidBody*>(b);
    if (!ra || !rb) return {};

    ShapeType sa = ra->getShapeType();
    ShapeType sb = rb->getShapeType();

    // Ground plane interactions (plane is always body B)
    auto isPlane = [](ShapeType s) { return s == ShapeType::PLANE || s == ShapeType::GROUND_PLANE; };

    if (isPlane(sb)) {
        if (sa == ShapeType::SPHERE) return sphereVsPlane(ra, rb);
        if (sa == ShapeType::BOX)    return boxVsPlane(ra, rb);
        if (sa == ShapeType::CYLINDER) return cylinderVsPlane(ra, rb);
    }
    if (isPlane(sa)) {
        if (sb == ShapeType::SPHERE) return sphereVsPlane(rb, ra);
        if (sb == ShapeType::BOX)    return boxVsPlane(rb, ra);
        if (sb == ShapeType::CYLINDER) return cylinderVsPlane(rb, ra);
    }

    // Sphere combinations
    if (sa == ShapeType::SPHERE && sb == ShapeType::SPHERE)
        return sphereVsSphere(ra, rb);
    if (sa == ShapeType::SPHERE && sb == ShapeType::BOX)
        return sphereVsBox(ra, rb);
    if (sa == ShapeType::BOX && sb == ShapeType::SPHERE)
        return sphereVsBox(rb, ra);
    if (sa == ShapeType::SPHERE && sb == ShapeType::CYLINDER)
        return sphereVsCylinder(ra, rb);
    if (sa == ShapeType::CYLINDER && sb == ShapeType::SPHERE)
        return sphereVsCylinder(rb, ra);

    // Fallback: bounding sphere
    return boundingSphereFallback(ra, rb);
}

// ---- Sphere vs Plane (plane normal = local Y-axis) ----
std::vector<ContactPair> CollisionDetector::sphereVsPlane(
    RigidBody* sphere, RigidBody* plane) const {

    double radius = sphere->getShape().dimensions.radius;
    Vec3 center = sphere->position;
    Vec3 planePos = plane->position;
    Vec3 planeNormal = plane->bodyToWorldDirection(Vec3(0, 1, 0));

    double dist = planeNormal.dot(center - planePos);
    double pen = radius - dist;

    if (pen < -threshold_) return {};

    ContactPair cp;
    cp.bodyA = sphere;
    cp.bodyB = plane;
    cp.normal = planeNormal;
    cp.penetration = std::max(0.0, pen);
    cp.pointA = center - planeNormal * radius;
    cp.pointB = center - planeNormal * dist;
    return {cp};
}

// ---- Sphere vs Sphere ----
std::vector<ContactPair> CollisionDetector::sphereVsSphere(
    RigidBody* a, RigidBody* b) const {

    double ra = a->getShape().dimensions.radius;
    double rb = b->getShape().dimensions.radius;
    Vec3 diff = b->position - a->position;
    double dist = diff.length();
    double pen = (ra + rb) - dist;

    if (pen < -threshold_) return {};

    Vec3 normal = dist > 1e-10 ? diff * (1.0 / dist) : Vec3(0, 1, 0);

    ContactPair cp;
    cp.bodyA = a;
    cp.bodyB = b;
    cp.normal = normal;
    cp.penetration = std::max(0.0, pen);
    cp.pointA = a->position + normal * ra;
    cp.pointB = b->position - normal * rb;
    return {cp};
}

// ---- Box vs Plane ----
std::vector<ContactPair> CollisionDetector::boxVsPlane(
    RigidBody* box, RigidBody* plane) const {

    const auto& dims = box->getShape().dimensions;
    double hx = dims.width * 0.5;
    double hy = dims.height * 0.5;
    double hz = dims.depth * 0.5;

    Vec3 planePos = plane->position;
    Vec3 planeNormal = plane->bodyToWorldDirection(Vec3(0, 1, 0));

    // 8 corners of the box in local coords
    Vec3 corners[8] = {
        {-hx, -hy, -hz}, { hx, -hy, -hz},
        {-hx,  hy, -hz}, { hx,  hy, -hz},
        {-hx, -hy,  hz}, { hx, -hy,  hz},
        {-hx,  hy,  hz}, { hx,  hy,  hz}
    };

    std::vector<ContactPair> contacts;
    for (auto& c : corners) {
        Vec3 worldCorner = box->bodyToWorld(c);
        double dist = planeNormal.dot(worldCorner - planePos);
        if (dist < threshold_) {
            ContactPair cp;
            cp.bodyA = box;
            cp.bodyB = plane;
            cp.normal = planeNormal;
            cp.penetration = std::max(0.0, -dist);
            cp.pointA = worldCorner;
            cp.pointB = worldCorner + planeNormal * (-dist);
            contacts.push_back(cp);
        }
    }
    return contacts;
}

// ---- Sphere vs Box (closest point on OBB) ----
std::vector<ContactPair> CollisionDetector::sphereVsBox(
    RigidBody* sphere, RigidBody* box) const {

    double r = sphere->getShape().dimensions.radius;
    Vec3 center = box->worldToBody(sphere->position);

    const auto& dims = box->getShape().dimensions;
    double hx = dims.width * 0.5;
    double hy = dims.height * 0.5;
    double hz = dims.depth * 0.5;

    // Clamp sphere center to box extents
    Vec3 closest(
        std::max(-hx, std::min(hx, center.x)),
        std::max(-hy, std::min(hy, center.y)),
        std::max(-hz, std::min(hz, center.z))
    );

    Vec3 diff = center - closest;
    double dist = diff.length();
    double pen = r - dist;

    if (pen < -threshold_) return {};

    Vec3 normal = dist > 1e-10 ? diff * (1.0 / dist) : Vec3(0, 1, 0);
    Vec3 worldNormal = box->bodyToWorldDirection(normal);
    Vec3 worldClosest = box->bodyToWorld(closest);

    ContactPair cp;
    cp.bodyA = sphere;
    cp.bodyB = box;
    cp.normal = worldNormal;
    cp.penetration = std::max(0.0, pen);
    cp.pointA = sphere->position - worldNormal * r;
    cp.pointB = worldClosest;
    return {cp};
}

// ---- Cylinder vs Plane ----
std::vector<ContactPair> CollisionDetector::cylinderVsPlane(
    RigidBody* cyl, RigidBody* plane) const {

    const auto& dims = cyl->getShape().dimensions;
    double radius = dims.radius;
    double halfH = dims.height * 0.5;

    Vec3 planePos = plane->position;
    Vec3 planeNormal = plane->bodyToWorldDirection(Vec3(0, 1, 0));

    // Cylinder axis in world = local Y
    Vec3 axis = cyl->bodyToWorldDirection(Vec3(0, 1, 0));

    // Bottom and top center
    Vec3 bottom = cyl->position - axis * halfH;
    Vec3 top = cyl->position + axis * halfH;

    // Project bottom circle edge closest to plane
    Vec3 edgeDir = planeNormal - axis * axis.dot(planeNormal);
    double edgeDirLen = edgeDir.length();
    if (edgeDirLen > 1e-10) edgeDir = edgeDir * (1.0 / edgeDirLen);
    else edgeDir = Vec3(0, 0, 0);

    std::vector<ContactPair> contacts;
    Vec3 candidates[2] = {
        bottom - edgeDir * radius,
        top - edgeDir * radius
    };

    for (auto& pt : candidates) {
        double dist = planeNormal.dot(pt - planePos);
        if (dist < threshold_) {
            ContactPair cp;
            cp.bodyA = cyl;
            cp.bodyB = plane;
            cp.normal = planeNormal;
            cp.penetration = std::max(0.0, -dist);
            cp.pointA = pt;
            cp.pointB = pt + planeNormal * (-dist);
            contacts.push_back(cp);
        }
    }
    return contacts;
}

// ---- Sphere vs Cylinder ----
std::vector<ContactPair> CollisionDetector::sphereVsCylinder(
    RigidBody* sphere, RigidBody* cyl) const {

    double sr = sphere->getShape().dimensions.radius;
    const auto& dims = cyl->getShape().dimensions;
    double cr = dims.radius;
    double halfH = dims.height * 0.5;

    Vec3 local = cyl->worldToBody(sphere->position);

    // Clamp to cylinder extent
    double clampedY = std::max(-halfH, std::min(halfH, local.y));
    Vec3 closest(local.x, clampedY, local.z);

    // Radial distance
    double radDist = std::sqrt(local.x * local.x + local.z * local.z);
    if (radDist > cr) {
        double scale = cr / radDist;
        closest.x = local.x * scale;
        closest.z = local.z * scale;
    }

    Vec3 diff = local - closest;
    double dist = diff.length();
    double pen = sr - dist;

    if (pen < -threshold_) return {};

    Vec3 normal = dist > 1e-10 ? diff * (1.0 / dist) : Vec3(0, 1, 0);
    Vec3 worldNormal = cyl->bodyToWorldDirection(normal);
    Vec3 worldClosest = cyl->bodyToWorld(closest);

    ContactPair cp;
    cp.bodyA = sphere;
    cp.bodyB = cyl;
    cp.normal = worldNormal;
    cp.penetration = std::max(0.0, pen);
    cp.pointA = sphere->position - worldNormal * sr;
    cp.pointB = worldClosest;
    return {cp};
}

// ---- Bounding sphere fallback ----
std::vector<ContactPair> CollisionDetector::boundingSphereFallback(
    RigidBody* a, RigidBody* b) const {

    double ra = getBoundingRadius(a);
    double rb = getBoundingRadius(b);

    Vec3 diff = b->position - a->position;
    double dist = diff.length();
    double pen = (ra + rb) - dist;

    if (pen < -threshold_) return {};

    Vec3 normal = dist > 1e-10 ? diff * (1.0 / dist) : Vec3(0, 1, 0);

    ContactPair cp;
    cp.bodyA = a;
    cp.bodyB = b;
    cp.normal = normal;
    cp.penetration = std::max(0.0, pen);
    cp.pointA = a->position + normal * ra;
    cp.pointB = b->position - normal * rb;
    return {cp};
}

double CollisionDetector::getBoundingRadius(RigidBody* body) const {
    const auto& d = body->getShape().dimensions;
    switch (body->getShapeType()) {
        case ShapeType::SPHERE: return d.radius;
        case ShapeType::BOX:
            return 0.5 * std::sqrt(d.width*d.width + d.height*d.height + d.depth*d.depth);
        case ShapeType::CYLINDER:
            return std::sqrt(d.radius*d.radius + d.height*d.height*0.25);
        default: return 1.0;
    }
}

} // namespace mb
