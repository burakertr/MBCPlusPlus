#pragma once
#include "mb/core/RigidBody.h"
#include "mb/forces/Force.h"
#include "mb/forces/SpringDamper.h"
#include "mb/constraints/PrismaticJoint.h"
#include "mb/constraints/RevoluteJoint.h"
#include "mb/constraints/FixedJoint.h"
#include "mb/constraints/Constraint.h"
#include "mb/vehicle/Wheel.h"
#include <memory>
#include <string>

namespace mb {

class MultibodySystem;

/**
 * Coil spring-damper element with optional preload.
 *   F_spring = k * (L0 - L) - preload   (compression positive)
 *   F_damper = -c * Ldot
 * Implemented as a thin wrapper around mb::SpringDamper for convenience and to
 * keep room for a non-linear stiffness curve later.
 */
struct CoilSpringDamperParams {
    double stiffness   = 30000.0;   // [N/m]
    double damping     = 3000.0;    // [N·s/m]
    double restLength  = 0.30;      // [m]
};

/**
 * Suspension corner parameters.
 *
 * Topology (per corner) — three bodies + three joints:
 *   chassis ──prismatic(Y)──> carrier ──revolute(Y)──> knuckle ──revolute(spinAxis)──> wheel
 *
 * Mounted at chassisAttachLocal in chassis frame. The corner is steered if
 * params.steerable is true (front corners). When non-steerable the carrier↔
 * knuckle revolute joint is heavily damped so it remains at θ = 0.
 */
struct SuspensionCornerParams {
    Vec3 chassisAttachLocal{0.0, 0.0, 0.0};   // [m] in chassis frame
    bool isLeftSide   = false;                 // affects spin direction
    bool steerable    = false;
    double carrierMass = 5.0;                  // [kg]
    double knuckleMass = 8.0;                  // [kg]
    double maxTravel   = 0.20;                 // [m] suspension stroke (informational)
    /// Static load this corner is expected to carry [N]. Used at attach time
    /// to pre-compress the spring so the system starts close to equilibrium
    /// (avoids violent transient at t=0). 0 disables preload.
    double designLoad  = 0.0;
    CoilSpringDamperParams springDamper;
    WheelParams wheel;
};

/**
 * One suspension corner (carrier + knuckle + wheel + joints + spring-damper).
 * Owns the bodies/constraints/forces but does NOT add them to a system; that
 * is done via attachToSystem().
 */
class SuspensionCorner {
public:
    SuspensionCorner(const std::string& name,
                     const SuspensionCornerParams& params,
                     std::shared_ptr<RigidBody> chassis,
                     std::shared_ptr<TireModel> tireModel);

    /// Add all owned bodies, constraints and forces to a MultibodySystem.
    /// Also creates and registers the TireForceElement for the wheel.
    /// Must be called after the chassis pose is finalized so initial poses
    /// are computed correctly.
    void attachToSystem(MultibodySystem& sys, const class Road& road);

    const std::string& name() const { return name_; }
    const SuspensionCornerParams& params() const { return params_; }

    std::shared_ptr<RigidBody> chassis() const { return chassis_; }
    std::shared_ptr<RigidBody> carrier() const { return carrier_; }
    std::shared_ptr<RigidBody> knuckle() const { return knuckle_; }
    Wheel& wheel() { return *wheel_; }
    const Wheel& wheel() const { return *wheel_; }

    /// Set the steering target angle (rad). Has effect only if steerable.
    /// Realised by a PD steering torque actuator.
    void setSteerTarget(double angleRad) { steerTarget_ = angleRad; }
    double steerTarget() const { return steerTarget_; }

    /// Current relative steer angle of knuckle vs carrier about the kingpin
    /// (chassis-Y in carrier frame), measured from the wheel's forward axis.
    double currentSteerAngle() const;

    /// Current vertical compression of the suspension spring (positive = compressed).
    double currentCompression() const;

    /// Spring + damper force currently applied (last sample, for diagnostics).
    double lastSpringForce() const { return lastSpringForce_; }

private:
    friend class SteeringActuator;
    friend class SuspensionDiagnostics;

    std::string name_;
    SuspensionCornerParams params_;
    std::shared_ptr<RigidBody> chassis_;
    std::shared_ptr<RigidBody> carrier_;
    std::shared_ptr<RigidBody> knuckle_;
    std::unique_ptr<Wheel> wheel_;

    std::shared_ptr<PrismaticJoint> travelJoint_;
    // Kingpin: a RevoluteJoint for steerable corners (front), a FixedJoint
    // for non-steerable corners (rear). Storing as the base class lets us
    // pick the constraint kind at attach time without templating the corner.
    std::shared_ptr<Constraint>     steerJoint_;
    std::shared_ptr<RevoluteJoint>  spinJoint_;
    std::shared_ptr<SpringDamper>   spring_;

    double steerTarget_ = 0.0;
    double lastSpringForce_ = 0.0;
};

/**
 * PD steering actuator: applies torque between knuckle and carrier about the
 * kingpin axis (chassis-Y in carrier frame) to drive the relative steer angle
 * to the corner's steerTarget(). Replaces a real steering rack for now or runs
 * alongside one as compliance.
 */
class SteeringActuator : public Force {
public:
    SteeringActuator(SuspensionCorner* corner, double kp, double kd,
                     double maxTorque = 2000.0,
                     const std::string& name = "SteeringActuator");
    void apply(double t) override;

private:
    SuspensionCorner* corner_;
    double kp_, kd_, maxTorque_;
};

/**
 * Anti-roll bar between two suspension corners (left & right of an axle).
 * Applies a force pair on the two carriers proportional to their suspension
 * compression difference, opposing roll.
 *   F = k_arb * (zL - zR)    on the more-compressed side: pushes up; opposite on the other.
 * Compressions are read from the prismatic joint travel.
 */
class AntiRollBar : public Force {
public:
    AntiRollBar(SuspensionCorner* left, SuspensionCorner* right,
                double stiffness,
                const std::string& name = "AntiRollBar");
    void apply(double t) override;

private:
    SuspensionCorner* left_;
    SuspensionCorner* right_;
    double k_;
};

} // namespace mb
