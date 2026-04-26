#include "mb/vehicle/Steering.h"
#include "mb/system/MultibodySystem.h"
#include <algorithm>
#include <cmath>

namespace mb {

Steering::Steering(SuspensionCorner* frontLeft, SuspensionCorner* frontRight,
                   SuspensionCorner* rearLeft,  SuspensionCorner* rearRight,
                   const SteeringParams& params)
    : frontLeft_(frontLeft), frontRight_(frontRight),
      rearLeft_(rearLeft), rearRight_(rearRight), params_(params) {}

void Steering::attachToSystem(MultibodySystem& sys) {
    actLeft_  = std::make_shared<SteeringActuator>(
        frontLeft_,  params_.actuator_kp, params_.actuator_kd,
        params_.actuator_max, "SteeringActuator_FL");
    actRight_ = std::make_shared<SteeringActuator>(
        frontRight_, params_.actuator_kp, params_.actuator_kd,
        params_.actuator_max, "SteeringActuator_FR");
    sys.addForce(actLeft_);
    sys.addForce(actRight_);

    // Rear corners are non-steerable and now use a FixedJoint kingpin
    // (see Suspension.cpp). No PD lock actuator is needed; the constraint
    // enforces zero relative steer kinematically.
}

void Steering::setSteeringWheelAngle(double rad) {
    steeringWheel_ = rad;
    double avgRoad = rad / std::max(params_.steeringRatio, 1e-6);
    avgRoad = std::max(-params_.maxRoadAngle,
                       std::min(params_.maxRoadAngle, avgRoad));

    if (std::abs(avgRoad) < 1e-5) {
        if (frontLeft_)  frontLeft_->setSteerTarget(0.0);
        if (frontRight_) frontRight_->setSteerTarget(0.0);
        return;
    }

    // Ackermann: compute turn radius from average angle, then per-wheel angles.
    //   tan(δ_avg) = L / R       =>  R = L / tan(δ_avg)
    //   δ_inner = atan( L / (R - t/2) ),  δ_outer = atan( L / (R + t/2) )
    const double L = std::max(params_.wheelbase, 1e-3);
    const double t = std::max(params_.track,     1e-3);
    const double R = L / std::tan(avgRoad);
    const double sgn = (avgRoad > 0.0) ? 1.0 : -1.0;
    const double absR = std::abs(R);
    double dInner = std::atan(L / std::max(absR - 0.5 * t, 1e-3));
    double dOuter = std::atan(L / std::max(absR + 0.5 * t, 1e-3));
    dInner *= sgn;
    dOuter *= sgn;

    // For positive avgRoad (turning right in our convention), the right wheel
    // is the inner one.
    if (avgRoad > 0.0) {
        if (frontRight_) frontRight_->setSteerTarget(dInner);
        if (frontLeft_)  frontLeft_->setSteerTarget(dOuter);
    } else {
        if (frontLeft_)  frontLeft_->setSteerTarget(dInner);
        if (frontRight_) frontRight_->setSteerTarget(dOuter);
    }
}

double Steering::currentRoadAngle() const {
    double a = 0.0;
    int n = 0;
    if (frontLeft_)  { a += frontLeft_->steerTarget();  ++n; }
    if (frontRight_) { a += frontRight_->steerTarget(); ++n; }
    return n > 0 ? a / n : 0.0;
}

} // namespace mb
