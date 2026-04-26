#pragma once

namespace mb {

class Vehicle;

/**
 * Driver input signals.
 *  throttle ∈ [-1, 1]  (negative = reverse for IdealMotor)
 *  brake    ∈ [ 0, 1]
 *  steering ∈ [-1, 1]  (mapped to steering wheel angle by Vehicle)
 */
struct DriverInput {
    double throttle = 0.0;
    double brake    = 0.0;
    double steering = 0.0;
};

/**
 * Abstract driver/controller. update() is called every Vehicle::update step
 * and may inspect the vehicle's state to compute new inputs.
 */
class Driver {
public:
    virtual ~Driver() = default;
    virtual DriverInput update(double t, double dt, const Vehicle& vehicle) = 0;
};

/**
 * Manual driver: returns whatever inputs the user sets externally
 * (e.g. from keyboard handling in a Qt demo).
 */
class ManualDriver : public Driver {
public:
    DriverInput update(double, double, const Vehicle&) override { return input_; }
    void setInput(const DriverInput& in) { input_ = in; }
    DriverInput& input() { return input_; }
private:
    DriverInput input_;
};

/**
 * Constant-speed driver: simple PI controller on longitudinal speed.
 * Steering input is passed through (set externally).
 */
class ConstantSpeedDriver : public Driver {
public:
    explicit ConstantSpeedDriver(double targetSpeed_mps,
                                 double kp = 0.4, double ki = 0.05)
        : target_(targetSpeed_mps), kp_(kp), ki_(ki) {}
    DriverInput update(double t, double dt, const Vehicle& vehicle) override;
    void setTargetSpeed(double v) { target_ = v; }
    void setSteering(double s) { steering_ = s; }

private:
    double target_;
    double kp_, ki_;
    double integ_ = 0.0;
    double steering_ = 0.0;
};

} // namespace mb
