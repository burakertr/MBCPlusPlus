#pragma once
#include "mb/vehicle/Wheel.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mb {

/**
 * Engine / motor interface: torque as a function of speed and throttle.
 * Speed is in rad/s at the engine output shaft (post-gearbox in our simple
 * model). Throttle ∈ [0, 1].
 */
class Engine {
public:
    virtual ~Engine() = default;
    virtual double torque(double omega_rad_s, double throttle) const = 0;
    virtual std::string name() const { return "Engine"; }
};

/**
 * Constant-power-limited ideal motor:
 *   torque = min(maxTorque, peakPower / max(|ω|, ω0)) * throttle
 * Direction follows throttle sign (negative throttle → reverse).
 */
class IdealMotor : public Engine {
public:
    IdealMotor(double maxTorque = 250.0, double peakPower = 100000.0)
        : maxTorque_(maxTorque), peakPower_(peakPower) {}
    double torque(double omega, double throttle) const override;
    std::string name() const override { return "IdealMotor"; }

private:
    double maxTorque_;
    double peakPower_;
};

/**
 * Open differential. Splits input torque equally between two output wheels.
 * Output wheel angular velocities are independent (each obeys its own
 * dynamics); the differential simply forwards torque.
 */
struct OpenDifferential {
    double finalDriveRatio = 3.5;   ///< overall ratio engine→wheel
};

/**
 * Drive type: which wheels receive engine torque.
 */
enum class DriveLayout {
    REAR_WHEEL_DRIVE,
    FRONT_WHEEL_DRIVE,
    ALL_WHEEL_DRIVE   // 50/50 front/rear, then per-axle open diff
};

struct DrivetrainParams {
    DriveLayout layout = DriveLayout::REAR_WHEEL_DRIVE;
    OpenDifferential diff;
    // AWD only: front:rear bias [0..1], 0 = full rear, 1 = full front.
    double awdFrontBias = 0.4;
    // Approximate engine output shaft inertia (reflected to wheels via ratio²).
    // Kept here for future use; unused in the pure-torque-pass-through path.
    double engineInertia = 0.10;
};

/**
 * High-level driveline: takes throttle, produces a target torque from the
 * engine, and distributes it to driven wheels via the differential and final
 * drive. Updated externally each step (Vehicle::update).
 */
class Drivetrain {
public:
    Drivetrain(std::shared_ptr<Engine> engine,
               const DrivetrainParams& params,
               Wheel* fl, Wheel* fr, Wheel* rl, Wheel* rr);

    /// Set throttle ∈ [-1, 1] and recompute per-wheel drive torques.
    /// Engine speed is taken as the average spin of driven wheels times the
    /// final-drive ratio.
    void update(double throttle);

    double engineSpeed() const { return engineSpeed_; }
    double engineTorque() const { return engineTorque_; }
    const DrivetrainParams& params() const { return params_; }

private:
    std::shared_ptr<Engine> engine_;
    DrivetrainParams params_;
    Wheel *fl_, *fr_, *rl_, *rr_;
    double engineSpeed_ = 0.0;
    double engineTorque_ = 0.0;
};

/**
 * Brake parameters: total max torque, front:rear bias.
 *   front share = bias, rear share = 1 - bias.
 * Each per-axle torque is split equally between left and right wheels.
 */
struct BrakeParams {
    double maxTotalTorque = 4000.0; ///< [N·m] full pedal
    double frontBias      = 0.65;   ///< [0..1]
};

class BrakeSystem {
public:
    BrakeSystem(const BrakeParams& params,
                Wheel* fl, Wheel* fr, Wheel* rl, Wheel* rr)
        : params_(params), fl_(fl), fr_(fr), rl_(rl), rr_(rr) {}

    /// Set brake pedal ∈ [0,1] and apply per-wheel brake torques.
    void update(double pedal);

    const BrakeParams& params() const { return params_; }

private:
    BrakeParams params_;
    Wheel *fl_, *fr_, *rl_, *rr_;
};

} // namespace mb
