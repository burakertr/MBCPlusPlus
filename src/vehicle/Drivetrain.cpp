#include "mb/vehicle/Drivetrain.h"
#include <algorithm>
#include <cmath>

namespace mb {

double IdealMotor::torque(double omega, double throttle) const {
    if (throttle == 0.0) return 0.0;
    const double aOmega = std::max(std::abs(omega), 5.0); // [rad/s] floor
    const double powerLimited = peakPower_ / aOmega;
    const double mag = std::min(maxTorque_, powerLimited);
    return std::copysign(mag * std::min(1.0, std::abs(throttle)), throttle);
}

// ---------------- Drivetrain ----------------

Drivetrain::Drivetrain(std::shared_ptr<Engine> engine,
                       const DrivetrainParams& params,
                       Wheel* fl, Wheel* fr, Wheel* rl, Wheel* rr)
    : engine_(std::move(engine)), params_(params),
      fl_(fl), fr_(fr), rl_(rl), rr_(rr) {}

void Drivetrain::update(double throttle) {
    throttle = std::max(-1.0, std::min(1.0, throttle));

    // Approximate driven-wheel average spin.
    auto axisSpin = [](Wheel* w) -> double {
        if (!w) return 0.0;
        Vec3 axis = w->body()->bodyToWorldDirection({0.0, 1.0, 0.0}).normalize();
        return w->body()->angularVelocity.dot(axis);
    };

    double wheelSpinAvg = 0.0;
    int n = 0;
    auto avgIfDriven = [&](bool driven, Wheel* w) {
        if (driven && w) { wheelSpinAvg += axisSpin(w); ++n; }
    };
    bool drvF = (params_.layout == DriveLayout::FRONT_WHEEL_DRIVE ||
                 params_.layout == DriveLayout::ALL_WHEEL_DRIVE);
    bool drvR = (params_.layout == DriveLayout::REAR_WHEEL_DRIVE ||
                 params_.layout == DriveLayout::ALL_WHEEL_DRIVE);
    avgIfDriven(drvF, fl_); avgIfDriven(drvF, fr_);
    avgIfDriven(drvR, rl_); avgIfDriven(drvR, rr_);
    if (n > 0) wheelSpinAvg /= n;

    // Engine speed = wheel speed × final-drive ratio.
    engineSpeed_  = wheelSpinAvg * params_.diff.finalDriveRatio;
    engineTorque_ = engine_ ? engine_->torque(engineSpeed_, throttle) : 0.0;

    // Wheel-side torque from engine through final drive (assume ideal, no losses).
    double T_wheel_total = engineTorque_ * params_.diff.finalDriveRatio;

    // Distribute per layout.
    auto setDrive = [](Wheel* w, double T) {
        if (w) w->setDriveTorque(T);
    };
    // Default: clear all driven torques.
    if (fl_) fl_->setDriveTorque(0.0);
    if (fr_) fr_->setDriveTorque(0.0);
    if (rl_) rl_->setDriveTorque(0.0);
    if (rr_) rr_->setDriveTorque(0.0);

    if (params_.layout == DriveLayout::REAR_WHEEL_DRIVE) {
        setDrive(rl_, 0.5 * T_wheel_total);
        setDrive(rr_, 0.5 * T_wheel_total);
    } else if (params_.layout == DriveLayout::FRONT_WHEEL_DRIVE) {
        setDrive(fl_, 0.5 * T_wheel_total);
        setDrive(fr_, 0.5 * T_wheel_total);
    } else { // AWD
        double bias = std::max(0.0, std::min(1.0, params_.awdFrontBias));
        double Tf = bias * T_wheel_total;
        double Tr = (1.0 - bias) * T_wheel_total;
        setDrive(fl_, 0.5 * Tf); setDrive(fr_, 0.5 * Tf);
        setDrive(rl_, 0.5 * Tr); setDrive(rr_, 0.5 * Tr);
    }
}

// ---------------- BrakeSystem ----------------

void BrakeSystem::update(double pedal) {
    pedal = std::max(0.0, std::min(1.0, pedal));
    const double Ttotal = pedal * params_.maxTotalTorque;
    const double bias   = std::max(0.0, std::min(1.0, params_.frontBias));
    const double Tf = 0.5 * bias        * Ttotal;
    const double Tr = 0.5 * (1.0 - bias) * Ttotal;
    if (fl_) fl_->setBrakeTorque(Tf);
    if (fr_) fr_->setBrakeTorque(Tf);
    if (rl_) rl_->setBrakeTorque(Tr);
    if (rr_) rr_->setBrakeTorque(Tr);
}

} // namespace mb
