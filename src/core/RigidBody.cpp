#include "mb/core/RigidBody.h"
#include <cmath>

namespace mb {

RigidBody::RigidBody(double mass, const Mat3& inertia, const std::string& name)
    : Body(BodyType::DYNAMIC, name), mass(mass), inertiaLocal(inertia) {}

std::shared_ptr<RigidBody> RigidBody::createBox(
    double mass, double sx, double sy, double sz, const std::string& name
) {
    double ix = mass / 12.0 * (sy*sy + sz*sz);
    double iy = mass / 12.0 * (sx*sx + sz*sz);
    double iz = mass / 12.0 * (sx*sx + sy*sy);
    auto body = std::make_shared<RigidBody>(mass, Mat3::diagonal(ix, iy, iz), name);
    Shape shape;
    shape.type = ShapeType::BOX;
    shape.position = Vec3::zero();
    shape.dimensions.halfExtents = Vec3(sx/2, sy/2, sz/2);
    body->shapes.push_back(shape);
    return body;
}

std::shared_ptr<RigidBody> RigidBody::createSphere(
    double mass, double radius, const std::string& name
) {
    double I = 2.0 / 5.0 * mass * radius * radius;
    auto body = std::make_shared<RigidBody>(mass, Mat3::diagonal(I, I, I), name);
    Shape shape;
    shape.type = ShapeType::SPHERE;
    shape.position = Vec3::zero();
    shape.dimensions.radius = radius;
    body->shapes.push_back(shape);
    return body;
}

std::shared_ptr<RigidBody> RigidBody::createCylinder(
    double mass, double radius, double height, const std::string& name
) {
    double iy = mass / 12.0 * (3*radius*radius + height*height);
    double iz = iy;
    double ix = mass / 2.0 * radius * radius;
    auto body = std::make_shared<RigidBody>(mass, Mat3::diagonal(iy, ix, iz), name);
    Shape shape;
    shape.type = ShapeType::CYLINDER;
    shape.position = Vec3::zero();
    shape.dimensions.radius = radius;
    shape.dimensions.height = height;
    body->shapes.push_back(shape);
    return body;
}

std::shared_ptr<RigidBody> RigidBody::createRod(
    double mass, double length, double radius, const std::string& name
) {
    return createCylinder(mass, radius, length, name);
}

std::shared_ptr<RigidBody> RigidBody::createGround(const std::string& name) {
    auto body = std::make_shared<RigidBody>(0.0, Mat3::zero(), name);
    body->type = BodyType::STATIC;
    body->mass = 0.0;
    Shape shape;
    shape.type = ShapeType::GROUND_PLANE;
    shape.position = Vec3::zero();
    body->shapes.push_back(shape);
    return body;
}

std::vector<double> RigidBody::getQ() const {
    return {position.x, position.y, position.z,
            orientation.w, orientation.x, orientation.y, orientation.z};
}

void RigidBody::setQ(const std::vector<double>& q) {
    if (q.size() >= 7) {
        position = Vec3(q[0], q[1], q[2]);
        orientation = Quaternion(q[3], q[4], q[5], q[6]);
    }
}

std::vector<double> RigidBody::getV() const {
    return {velocity.x, velocity.y, velocity.z,
            angularVelocity.x, angularVelocity.y, angularVelocity.z};
}

void RigidBody::setV(const std::vector<double>& v) {
    if (v.size() >= 6) {
        velocity = Vec3(v[0], v[1], v[2]);
        angularVelocity = Vec3(v[3], v[4], v[5]);
    }
}

std::vector<double> RigidBody::computeQDot() const {
    // q̇ = [v, 0.5 * G(q) * ω]
    Quaternion qDot = orientation.derivative(angularVelocity);
    return {velocity.x, velocity.y, velocity.z,
            qDot.w, qDot.x, qDot.y, qDot.z};
}

std::vector<double> RigidBody::computeMassBlock() const {
    // 6x6 block: diag(m,m,m) and inertia in world frame
    Mat3 Iworld = getInertiaWorld();
    std::vector<double> block(36, 0.0);
    block[0*6+0] = mass;
    block[1*6+1] = mass;
    block[2*6+2] = mass;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            block[(3+r)*6+(3+c)] = Iworld.get(r, c);
    return block;
}

std::vector<double> RigidBody::computeForces(const Vec3& gravity) {
    // Total force = accumulated external + gravity
    Vec3 gForce = Vec3(gravity.x * mass, gravity.y * mass, gravity.z * mass);
    Vec3 totalForce = accumulatedForce.add(gForce);

    // Gyroscopic torque: τ_gyro = -ω × (I_world * ω)
    Mat3 Iworld = getInertiaWorld();
    Vec3 Iomega = Iworld.multiplyVec3(angularVelocity);
    Vec3 gyro = angularVelocity.cross(Iomega).negate();
    Vec3 totalTorque = accumulatedTorque.add(gyro);

    return {totalForce.x, totalForce.y, totalForce.z,
            totalTorque.x, totalTorque.y, totalTorque.z};
}

double RigidBody::computeKineticEnergy() const {
    double translational = 0.5 * mass * velocity.lengthSquared();
    Mat3 Iworld = getInertiaWorld();
    Vec3 Iomega = Iworld.multiplyVec3(angularVelocity);
    double rotational = 0.5 * angularVelocity.dot(Iomega);
    return translational + rotational;
}

double RigidBody::computePotentialEnergy(const Vec3& gravity) const {
    // PE = -m * g · r (relative to origin)
    return -mass * gravity.dot(position);
}

Mat3 RigidBody::getInertiaWorld() const {
    Mat3 R = orientation.toRotationMatrix();
    return R.multiply(inertiaLocal).multiply(R.transpose());
}

Mat3 RigidBody::getInertiaInverseWorld() const {
    return getInertiaWorld().inverse();
}

std::vector<double> RigidBody::getMassMatrix() const {
    return computeMassBlock();
}

} // namespace mb
