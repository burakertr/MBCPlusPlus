#include "mb/vehicle/Driver.h"
#include "mb/vehicle/Vehicle.h"
#include <algorithm>

namespace mb {

DriverInput ConstantSpeedDriver::update(double /*t*/, double dt,
                                        const Vehicle& vehicle) {
    DriverInput out;
    double v = vehicle.forwardSpeed();
    double err = target_ - v;
    integ_ += err * dt;
    // Anti-windup clamp.
    integ_ = std::max(-50.0, std::min(50.0, integ_));
    double u = kp_ * err + ki_ * integ_;
    if (u >= 0.0) {
        out.throttle = std::min(1.0, u);
        out.brake    = 0.0;
    } else {
        out.throttle = 0.0;
        out.brake    = std::min(1.0, -u);
    }
    out.steering = std::max(-1.0, std::min(1.0, steering_));
    return out;
}

} // namespace mb
