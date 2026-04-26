#pragma once
#include "mb/vehicle/Suspension.h"
#include <memory>
#include <string>

namespace mb {

/**
 * Steering parameters. The steering "input" is the steering wheel angle [rad];
 * road-wheel angle = input / steeringRatio. Ackermann geometry is approximated
 * from wheelbase and track to give differential steer angles to the inner and
 * outer wheels in a turn.
 */
struct SteeringParams {
    double steeringRatio = 16.0;   // [-]
    double maxRoadAngle  = 0.6;    // [rad] hard limit on road wheel angle
    double wheelbase     = 2.7;    // [m]
    double track         = 1.55;   // [m]
    double actuator_kp   = 4000.0; // [N·m/rad]
    double actuator_kd   = 300.0;  // [N·m·s/rad]
    double actuator_max  = 2500.0; // [N·m]
    // PD lock holding the non-steerable (rear) corners at 0° steer. Stiffer
    // than the front actuator since these never move; counteracts tyre Mz
    // and any residual drivetrain reaction that reaches the kingpin.
    double rearLock_kp   = 8000.0; // [N·m/rad]
    double rearLock_kd   = 800.0;  // [N·m·s/rad]
    double rearLock_max  = 5000.0; // [N·m]
};

/**
 * Distributes a steering input to the front suspension corners' steer targets,
 * applying Ackermann geometry. Owns one SteeringActuator per front corner.
 */
class Steering {
public:
    Steering(SuspensionCorner* frontLeft, SuspensionCorner* frontRight,
             SuspensionCorner* rearLeft,  SuspensionCorner* rearRight,
             const SteeringParams& params);

    /// Register the per-corner PD steering actuators with the system.
    void attachToSystem(class MultibodySystem& sys);

    /// Set the steering wheel angle [rad]. Positive = turn right (per Bosch
    /// convention; chassis Y is up, so positive yaw is left in our right-handed
    /// frame: a positive steering input causes the vehicle to yaw negatively
    /// about Y -> turn right when X is forward).
    void setSteeringWheelAngle(double rad);

    double steeringWheelAngle() const { return steeringWheel_; }
    double currentRoadAngle() const;   ///< average of front corner targets

    const SteeringParams& params() const { return params_; }

private:
    SuspensionCorner* frontLeft_;
    SuspensionCorner* frontRight_;
    SuspensionCorner* rearLeft_;
    SuspensionCorner* rearRight_;
    SteeringParams params_;
    double steeringWheel_ = 0.0;

    std::shared_ptr<SteeringActuator> actLeft_, actRight_;
    std::shared_ptr<SteeringActuator> rearLockLeft_, rearLockRight_;
};

} // namespace mb
