#pragma once
#include "mb/core/RigidBody.h"
#include "mb/vehicle/Suspension.h"
#include "mb/vehicle/Steering.h"
#include "mb/vehicle/Drivetrain.h"
#include "mb/vehicle/Driver.h"
#include "mb/vehicle/Road.h"
#include "mb/vehicle/Wheel.h"
#include "mb/vehicle/TireModel.h"
#include <memory>
#include <string>
#include <array>
#include <functional>

namespace mb {

class MultibodySystem;

/**
 * Top-level parameter bundle for a 4-wheeled vehicle template (sedan).
 *
 * Frame convention (chassis local):
 *   +X = forward
 *   +Y = up
 *   +Z = right (right-handed: X × Y = Z)
 */
struct VehicleParams {
    std::string name = "Vehicle";

    // Chassis
    double chassisMass = 1500.0;     // [kg] (sprung mass approximation)
    Vec3   chassisHalfExtents{2.30, 0.40, 0.85}; // [m] half lengths along X,Y,Z
    Vec3   initialPosition{0.0, 0.6, 0.0};
    // Initial yaw [rad] about Y (positive yaw = +X rotates toward -Z, i.e. "left").

    double initialYaw = 0.0;

    // Geometry
    double wheelbase   = 2.70;       // [m]
    double track       = 1.55;       // [m]
    double cgHeight    = 0.55;       // [m] above ground (informational)
    double cgFrontBias = 0.55;       // weight fraction on front axle (>0.5 → understeer)

    // Suspension (same params on all corners by default; modify after construction
    // for per-axle tuning).
    SuspensionCornerParams frontSusp;
    SuspensionCornerParams rearSusp;

    // Anti-roll bars (0 disables).
    double antiRollFront = 15000.0;  // [N/m]
    double antiRollRear  = 10000.0;  // [N/m]

    // Steering / drivetrain / brakes
    SteeringParams   steering;
    DrivetrainParams drivetrain;
    BrakeParams      brakes;

    // Tyre model factory (returns one fresh instance per wheel; if null, a
    // PacejkaTireModel with default params is used).
    std::function<std::shared_ptr<TireModel>()> tireFactory;

    // Aerodynamic-style drag and yaw damping applied to the chassis. These
    // model the (otherwise missing) speed-squared aero drag and the
    // stabilising yaw-rate damping that real cars get from a combination
    // of aero + compliance + suspension friction. Without them, a fully
    // gas-pedalled rigid-body car has no source of lateral / yaw damping
    // and any tiny perturbation grows exponentially after a few seconds.
    double aeroDragLin     = 30.0;   // [N·s/m]   linear part
    double aeroDragQuad    = 0.6;    // [N·s²/m²] quadratic part (~0.5·ρ·Cd·A)
    double yawRateDamping  = 25000.0; // [N·m·s/rad] yaw-rate damper
    double pitchRollDamping = 5000.0; // [N·m·s/rad] pitch & roll damper
};

/**
 * High-level orchestrator. Owns chassis + 4 corners + steering + drivetrain
 * + brakes + (optional) driver and integrates them with a MultibodySystem.
 */
class Vehicle {
public:
    explicit Vehicle(const VehicleParams& params = VehicleParams{});

    /// Add the vehicle's bodies, constraints and forces to a system. The road
    /// reference is captured (not copied) and used by the tire force elements.
    void attachToSystem(MultibodySystem& sys, const Road& road);

    /// Per-step update: query driver, push inputs through steering / drivetrain
    /// / brakes. Should be called once per integrator step BEFORE
    /// MultibodySystem::step (or inside its callback for sub-stepped Qt loops).
    void update(double t, double dt);

    /// Manual driver input (used when no Driver is attached).
    void setManualInput(const DriverInput& in) { manualInput_ = in; }
    void setDriver(std::shared_ptr<Driver> d) { driver_ = std::move(d); }

    // ---- Diagnostics / accessors ----
    std::shared_ptr<RigidBody> chassis() const { return chassis_; }
    SuspensionCorner& corner(int idx) { return *corners_[idx]; }   // 0=FL,1=FR,2=RL,3=RR
    const SuspensionCorner& corner(int idx) const { return *corners_[idx]; }

    Wheel& wheel(int idx) { return corners_[idx]->wheel(); }
    const Wheel& wheel(int idx) const { return corners_[idx]->wheel(); }

    Steering*   steering()   { return steering_.get(); }
    Drivetrain* drivetrain() { return drivetrain_.get(); }
    BrakeSystem* brakes()    { return brakes_.get(); }

    /// Forward speed of chassis (chassis-X projected onto world velocity) [m/s].
    double forwardSpeed() const;

    /// Lateral speed of chassis [m/s].
    double lateralSpeed() const;

    /// Yaw angle of the chassis about world-Y [rad].
    double yawAngle() const;

    const VehicleParams& params() const { return params_; }

    /// Convenience factory: a 4-wheel sedan with sensible defaults.
    static VehicleParams sedanDefaults(const std::string& name = "Sedan");

private:
    VehicleParams params_;
    std::shared_ptr<RigidBody> chassis_;
    // Corner indices: 0 = FL, 1 = FR, 2 = RL, 3 = RR.
    std::array<std::unique_ptr<SuspensionCorner>, 4> corners_;
    std::unique_ptr<Steering>    steering_;
    std::unique_ptr<Drivetrain>  drivetrain_;
    std::unique_ptr<BrakeSystem> brakes_;
    std::shared_ptr<AntiRollBar> arbFront_, arbRear_;

    std::shared_ptr<Driver> driver_;
    DriverInput manualInput_;

    // Cached pointer to the road set in attachToSystem(). Used by update()
    // to refresh per-wheel diagnostics from the converged state once per
    // step (independent of the integrator's many internal probe calls).
    const class Road* road_ = nullptr;

    bool attached_ = false;
};

} // namespace mb
