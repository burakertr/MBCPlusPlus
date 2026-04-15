#pragma once
#include "mb/math/Vec3.h"
#include "mb/math/Mat3.h"
#include "mb/math/Quaternion.h"
#include <vector>
#include <string>
#include <memory>

namespace mb {

enum class BodyType {
    DYNAMIC,
    KINEMATIC,
    STATIC
};

struct MassProperties {
    double mass = 1.0;
    Mat3 inertia = Mat3::identity();
    Vec3 centerOfMass = Vec3::zero();
};

struct BodyState {
    Vec3 position;
    Vec3 velocity;
    Quaternion orientation;
    Vec3 angularVelocity;
};

/**
 * Abstract base class for bodies in the multibody system
 */
class Body {
public:
    int id;
    std::string name;
    BodyType type;

    Body(BodyType type = BodyType::DYNAMIC, const std::string& name = "");
    virtual ~Body() = default;

    // State accessors
    Vec3 position;
    Vec3 velocity;
    Quaternion orientation;
    Vec3 angularVelocity;

    // Force accumulation
    Vec3 accumulatedForce;
    Vec3 accumulatedTorque;

    void clearForces();
    void applyForce(const Vec3& force);
    void applyTorque(const Vec3& torque);
    void applyForceAtPoint(const Vec3& force, const Vec3& worldPoint);

    // Coordinate transformations
    Vec3 bodyToWorld(const Vec3& localPoint) const;
    Vec3 worldToBody(const Vec3& worldPoint) const;
    Vec3 bodyToWorldDirection(const Vec3& localDir) const;
    Vec3 worldToBodyDirection(const Vec3& worldDir) const;

    bool isDynamic() const { return type == BodyType::DYNAMIC; }
    bool isStatic() const { return type == BodyType::STATIC; }

    // Virtual interface
    virtual double getMass() const = 0;
    virtual int nq() const = 0; // number of position coordinates
    virtual int nv() const = 0; // number of velocity coordinates

    virtual std::vector<double> getQ() const = 0;
    virtual void setQ(const std::vector<double>& q) = 0;
    virtual std::vector<double> getV() const = 0;
    virtual void setV(const std::vector<double>& v) = 0;

    virtual std::vector<double> computeQDot() const = 0;
    virtual std::vector<double> computeMassBlock() const = 0;
    virtual std::vector<double> computeForces(const Vec3& gravity) = 0;

    virtual double computeKineticEnergy() const = 0;
    virtual double computePotentialEnergy(const Vec3& gravity) const = 0;

    virtual Vec3 getPointVelocity(const Vec3& worldPoint) const;

private:
    static int nextId_;
};

} // namespace mb
