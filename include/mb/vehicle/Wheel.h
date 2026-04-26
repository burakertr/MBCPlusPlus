#pragma once
#include "mb/forces/Force.h"
#include "mb/core/RigidBody.h"
#include "mb/vehicle/TireModel.h"
#include "mb/vehicle/Road.h"
#include <memory>
#include <string>

namespace mb {

/**
 * Wheel parameters (rigid wheel + analytic vertical contact + slip-based tyre).
 */
struct WheelParams {
    double mass     = 20.0;     // [kg]
    double radius   = 0.31;     // [m]
    double width    = 0.20;     // [m] (cylinder height)
    // Vertical contact: F_z = max(0, kz*delta + cz*delta_dot)
    double kz       = 2.0e5;    // [N/m]
    double cz       = 2.0e3;    // [N·s/m]
    // Numerical slip-velocity guard (avoids singular kappa at low speed).
    // Acts as the denominator for kappa = (Vroll - Vx)/Vguard whenever the
    // forward speed falls below this threshold. Larger values are more
    // robust at low/zero speed at the cost of slightly under-predicted slip
    // near the threshold (the tyre still transitions smoothly to the full
    // Pacejka response above).
    double vEps     = 10.0;     // [m/s]
    // Optional rolling resistance coefficient (longitudinal, opposes motion).
    double rolling_resistance = 0.012;
};

/**
 * A rigid wheel attached to a knuckle (or chassis) via a revolute spin joint
 * created by the suspension. The Wheel object owns the wheel RigidBody and a
 * TireModel; its own contribution to the dynamics is delivered through a
 * Force element (TireForceElement) added to the MultibodySystem.
 *
 * Spin axis convention: wheel local +Y is the spin axis. The body is created
 * via RigidBody::createCylinder(mass, radius, width), so the cylinder height
 * matches the wheel width and Iyy is the polar (spin) inertia.
 */
class Wheel {
public:
    Wheel(const std::string& name,
          const WheelParams& params,
          std::shared_ptr<TireModel> tireModel);

    const std::string& name() const { return name_; }
    const WheelParams& params() const { return params_; }
    std::shared_ptr<RigidBody> body() const { return body_; }
    std::shared_ptr<TireModel> tireModel() const { return tireModel_; }

    /// Optional pointer to the knuckle (steering carrier) the wheel is mounted
    /// on via the spin joint. Used by TireForceElement to (a) project drive /
    /// brake / self-aligning torques onto the knuckle's spin axis instead of
    /// the wheel's instantaneous spin axis (the latter rotates with the wheel
    /// and any tiny camber/steer drift would leak torque into the kingpin
    /// and lateral DOFs, causing self-steer and chassis yaw drift) and
    /// (b) apply the equal-and-opposite reaction torques to the knuckle so
    /// the drivetrain reaction reaches the chassis through the joint chain
    /// (Newton's 3rd law). If null, torques are applied to the wheel only
    /// using its own instantaneous axis (legacy behaviour).
    void setKnuckle(RigidBody* knuckle) { knuckle_ = knuckle; }
    RigidBody* knuckle() const { return knuckle_; }

    // Per-step actuator inputs (Nm). Sign convention: drive torque accelerates
    // the wheel about its spin axis; brake torque opposes spin (sign handled
    // automatically inside the force element).
    void setDriveTorque(double Nm) { driveTorque_ = Nm; }
    void setBrakeTorque(double Nm) { brakeTorque_ = Nm < 0.0 ? 0.0 : Nm; }
    double driveTorque() const { return driveTorque_; }
    double brakeTorque() const { return brakeTorque_; }

    // Diagnostics from last force evaluation.
    double lastFz()    const { return lastFz_; }
    double lastKappa() const { return lastKappa_; }
    double lastAlpha() const { return lastAlpha_; }
    double lastSpin()  const { return lastSpin_; }   // [rad/s]
    bool   lastInContact() const { return lastInContact_; }

    /// Recompute and store diagnostic values (Fz, kappa, alpha, spin,
    /// in-contact) from the current converged kinematic state, WITHOUT
    /// applying any forces or torques. Use this to refresh the lastXxx()
    /// readouts once per macro time-step (e.g. from Vehicle::update) so
    /// they are not contaminated by the implicit integrator's intermediate
    /// finite-difference Jacobian probes — which call the Force apply()
    /// chain many times per step with perturbed velocities.
    void sampleDiagnostics(const Road& road);

private:
    friend class TireForceElement;

    /// Internal: evaluate contact + tyre model. When applyForces=true the
    /// resulting forces and torques are applied to the wheel and reacted on
    /// the knuckle. When false, only the lastXxx_ diagnostic fields are
    /// updated. Used by both TireForceElement::apply and sampleDiagnostics.
    void evaluateContact_(const Road& road, bool applyForces);

    std::string name_;
    WheelParams params_;
    std::shared_ptr<RigidBody> body_;
    std::shared_ptr<TireModel> tireModel_;
    RigidBody* knuckle_ = nullptr;

    double driveTorque_ = 0.0;
    double brakeTorque_ = 0.0;

    // Diagnostics (written by TireForceElement).
    double lastFz_    = 0.0;
    double lastKappa_ = 0.0;
    double lastAlpha_ = 0.0;
    double lastSpin_  = 0.0;
    bool   lastInContact_ = false;
};

/**
 * Force element coupling a Wheel to a Road. Computes vertical normal load
 * from analytic penetration, slip from contact-patch velocity, calls the
 * tyre model and applies (Fx, Fy, Mz) plus drive/brake torque to the wheel.
 */
class TireForceElement : public Force {
public:
    TireForceElement(Wheel* wheel, const Road* road,
                     const std::string& name = "TireForce");

    void apply(double t) override;

private:
    Wheel* wheel_;
    const Road* road_;
};

} // namespace mb
