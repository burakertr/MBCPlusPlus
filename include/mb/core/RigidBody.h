#pragma once
#include "Body.h"
#include "mb/math/Vec3.h"
#include "mb/math/Mat3.h"
#include "mb/math/Quaternion.h"
#include <vector>
#include <string>

namespace mb {

enum class ShapeType {
    SPHERE,
    BOX,
    CYLINDER,
    CONE,
    MESH,
    PLANE,
    GROUND_PLANE
};

struct ShapeDimensions {
    double radius = 0;
    double height = 0;
    double width = 0;
    double depth = 0;
    Vec3 halfExtents;
};

struct Shape {
    ShapeType type;
    Vec3 position;         // Local offset from body origin
    Quaternion orientation; // Local rotation
    ShapeDimensions dimensions;
    std::string meshUrl;
};

/**
 * Rigid body with mass, inertia, and collision shapes.
 * nq = 7 (position[3] + quaternion[4])
 * nv = 6 (velocity[3] + angularVelocity[3])
 */
class RigidBody : public Body {
public:
    double mass;
    Mat3 inertiaLocal; // Inertia in body frame
    std::vector<Shape> shapes;

    RigidBody(double mass = 1.0, const Mat3& inertia = Mat3::identity(),
              const std::string& name = "");

    // Factory methods
    static std::shared_ptr<RigidBody> createBox(
        double mass, double sx, double sy, double sz, const std::string& name = "Box");
    static std::shared_ptr<RigidBody> createSphere(
        double mass, double radius, const std::string& name = "Sphere");
    static std::shared_ptr<RigidBody> createCylinder(
        double mass, double radius, double height, const std::string& name = "Cylinder");
    static std::shared_ptr<RigidBody> createRod(
        double mass, double length, double radius, const std::string& name = "Rod");
    static std::shared_ptr<RigidBody> createGround(const std::string& name = "Ground");

    // Body interface
    double getMass() const override { return mass; }
    int nq() const override { return 7; }
    int nv() const override { return 6; }

    std::vector<double> getQ() const override;
    void setQ(const std::vector<double>& q) override;
    std::vector<double> getV() const override;
    void setV(const std::vector<double>& v) override;

    std::vector<double> computeQDot() const override;
    std::vector<double> computeMassBlock() const override;
    std::vector<double> computeForces(const Vec3& gravity) override;

    double computeKineticEnergy() const override;
    double computePotentialEnergy(const Vec3& gravity) const override;

    // Inertia in world frame
    Mat3 getInertiaWorld() const;
    Mat3 getInertiaInverseWorld() const;

    // Get mass matrix (6x6)
    std::vector<double> getMassMatrix() const;

    // Convenience: primary shape type (first shape, or SPHERE default)
    ShapeType getShapeType() const {
        return shapes.empty() ? ShapeType::SPHERE : shapes[0].type;
    }
    const Shape& getShape() const {
        static Shape defaultShape{ShapeType::SPHERE, Vec3::zero(), Quaternion::identity(), {1,0,{}}, ""};
        return shapes.empty() ? defaultShape : shapes[0];
    }
};

} // namespace mb
