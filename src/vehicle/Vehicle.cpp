#include "mb/vehicle/Vehicle.h"
#include "mb/system/MultibodySystem.h"
#include "mb/forces/Force.h"
#include <algorithm>
#include <cmath>

namespace mb {

namespace {

// Aerodynamic-style drag (linear + quadratic on body velocity, expressed in
// world frame) and rotational damping (yaw + pitch + roll) applied to the
// chassis. Rigid-body cars without aero or compliance damping are
// neutrally stable in the lateral/yaw direction; any tiny perturbation
// from the constraint solver will exponentially grow once the car has
// some forward speed. This force restores the small but essential
// damping that real-world vehicles get from aero, suspension friction
// and tyre-road compliance. Coefficients are intentionally modest so as
// not to alter the qualitative dynamics at low speed.
class ChassisAeroDamper : public Force {
public:
    ChassisAeroDamper(RigidBody* chassis,
                      double cLin, double cQuad,
                      double cYaw, double cPitchRoll,
                      const std::string& name)
        : Force(name), chassis_(chassis),
          cLin_(cLin), cQuad_(cQuad),
          cYaw_(cYaw), cPitchRoll_(cPitchRoll) {}

    void apply(double /*t*/) override {
        if (!chassis_) return;
        // Linear drag: F = -(cLin + cQuad·|v|) · v   (world frame).
        const Vec3& v = chassis_->velocity;
        double speed = v.length();
        double k = cLin_ + cQuad_ * speed;
        chassis_->applyForce(v.scale(-k));

        // Rotational damping in body frame: separate yaw vs pitch/roll.
        Vec3 wWorld = chassis_->angularVelocity;
        Vec3 wBody  = chassis_->orientation.conjugate().rotate(wWorld);
        Vec3 tauBody{
            -cPitchRoll_ * wBody.x,   // roll about chassis +X
            -cYaw_       * wBody.y,   // yaw  about chassis +Y
            -cPitchRoll_ * wBody.z    // pitch about chassis +Z
        };
        Vec3 tauWorld = chassis_->orientation.rotate(tauBody);
        chassis_->applyTorque(tauWorld);
    }

private:
    RigidBody* chassis_;
    double cLin_, cQuad_, cYaw_, cPitchRoll_;
};

} // namespace

VehicleParams Vehicle::sedanDefaults(const std::string& name) {
    VehicleParams p;
    p.name = name;
    p.chassisMass = 1000.0;
    p.chassisHalfExtents = {2.30, 0.40, 0.85};
    p.wheelbase = 2.70;
    p.track     = 1.55;
    p.cgHeight  = 0.55;
    p.cgFrontBias = 0.55;

    // Suspension defaults
    p.frontSusp.steerable = true;
    p.frontSusp.springDamper.stiffness  = 35000.0;
    p.frontSusp.springDamper.damping    = 3500.0;
    p.frontSusp.springDamper.restLength = 0.30;
    p.frontSusp.maxTravel = 0.20;
    p.frontSusp.wheel.mass   = 22.0;
    p.frontSusp.wheel.radius = 0.32;
    p.frontSusp.wheel.width  = 0.21;
    p.frontSusp.wheel.kz     = 2.5e5;
    p.frontSusp.wheel.cz     = 2.5e3;

    p.rearSusp = p.frontSusp;
    p.rearSusp.steerable = false;
    p.rearSusp.springDamper.stiffness = 32000.0;
    p.rearSusp.springDamper.damping   = 3200.0;

    // Steering / drivetrain / brakes
    p.steering.steeringRatio = 16.0;
    p.steering.maxRoadAngle  = 0.55;
    p.steering.wheelbase = p.wheelbase;
    p.steering.track     = p.track;

    p.drivetrain.layout = DriveLayout::REAR_WHEEL_DRIVE;
    p.drivetrain.diff.finalDriveRatio = 3.5;

    p.brakes.maxTotalTorque = 4000.0;
    p.brakes.frontBias      = 0.65;

    p.antiRollFront = 18000.0;
    p.antiRollRear  = 12000.0;

    return p;
}

Vehicle::Vehicle(const VehicleParams& params) : params_(params) {
    // Chassis.
    const auto& he = params_.chassisHalfExtents;
    chassis_ = RigidBody::createBox(params_.chassisMass,
                                    2.0 * he.x, 2.0 * he.y, 2.0 * he.z,
                                    params_.name + "_Chassis");
    chassis_->position = params_.initialPosition;
    chassis_->orientation = Quaternion::fromAxisAngle({0.0, 1.0, 0.0},
                                                      params_.initialYaw);

    // Tyre factory.
    if (!params_.tireFactory) {
        params_.tireFactory = []() {
            return std::make_shared<PacejkaTireModel>(PacejkaParams{});
        };
    }

    // Corner attach points in chassis frame.
    const double Lf = params_.wheelbase * (1.0 - params_.cgFrontBias);
    const double Lr = params_.wheelbase * params_.cgFrontBias;
    const double tHalf = 0.5 * params_.track;
    // Suspension top mount: a bit below chassis top, ON the wheel centreline.
    // Placing the kingpin axis through the wheel centre eliminates the
    // scrub-radius moment Fx·r_scrub that would otherwise self-steer the
    // corner toe-in under traction or rolling resistance.
    const double mountY = -he.y * 0.4;        // slightly below chassis CG
    const double mountZ = tHalf;              // wheel-centre kingpin (zero scrub)

    // Per-corner static weight to pre-compress the springs at startup.
    // Sprung mass only (chassis); unsprung settles the residual via tire spring.
    const double g = 9.81;
    const double WfPerCorner = 0.5 * params_.chassisMass * g * params_.cgFrontBias;
    const double WrPerCorner = 0.5 * params_.chassisMass * g * (1.0 - params_.cgFrontBias);

    auto makeCorner = [&](const std::string& tag, bool left, bool front,
                          double x, double z) -> std::unique_ptr<SuspensionCorner> {
        SuspensionCornerParams cp = front ? params_.frontSusp : params_.rearSusp;
        cp.chassisAttachLocal = Vec3(x, mountY, z);
        cp.isLeftSide = left;
        cp.steerable  = front && params_.frontSusp.steerable;
        cp.designLoad = front ? WfPerCorner : WrPerCorner;
        return std::make_unique<SuspensionCorner>(
            params_.name + "_" + tag, cp, chassis_, params_.tireFactory());
    };

    corners_[0] = makeCorner("FL", true,  true,   Lf, -mountZ);
    corners_[1] = makeCorner("FR", false, true,   Lf,  mountZ);
    corners_[2] = makeCorner("RL", true,  false, -Lr, -mountZ);
    corners_[3] = makeCorner("RR", false, false, -Lr,  mountZ);

    // Steering, drivetrain, brakes.
    steering_ = std::make_unique<Steering>(corners_[0].get(), corners_[1].get(),
                                           corners_[2].get(), corners_[3].get(),
                                           params_.steering);
    drivetrain_ = std::make_unique<Drivetrain>(
        std::make_shared<IdealMotor>(), params_.drivetrain,
        &corners_[0]->wheel(), &corners_[1]->wheel(),
        &corners_[2]->wheel(), &corners_[3]->wheel());
    brakes_ = std::make_unique<BrakeSystem>(
        params_.brakes,
        &corners_[0]->wheel(), &corners_[1]->wheel(),
        &corners_[2]->wheel(), &corners_[3]->wheel());
}

void Vehicle::attachToSystem(MultibodySystem& sys, const Road& road) {
    if (attached_) return;
    road_ = &road;
    sys.addBody(chassis_);
    for (auto& c : corners_) c->attachToSystem(sys, road);
    steering_->attachToSystem(sys);

    if (params_.antiRollFront > 0.0) {
        arbFront_ = std::make_shared<AntiRollBar>(corners_[0].get(),
                                                  corners_[1].get(),
                                                  params_.antiRollFront,
                                                  params_.name + "_ARB_F");
        sys.addForce(arbFront_);
    }
    if (params_.antiRollRear > 0.0) {
        arbRear_ = std::make_shared<AntiRollBar>(corners_[2].get(),
                                                 corners_[3].get(),
                                                 params_.antiRollRear,
                                                 params_.name + "_ARB_R");
        sys.addForce(arbRear_);
    }

    // Aerodynamic / yaw damping on the chassis (essential for stability of
    // a fully-throttled rigid-body car — see ChassisAeroDamper above).
    if (params_.aeroDragLin > 0.0 || params_.aeroDragQuad > 0.0 ||
        params_.yawRateDamping > 0.0 || params_.pitchRollDamping > 0.0) {
        sys.addForce(std::make_shared<ChassisAeroDamper>(
            chassis_.get(),
            params_.aeroDragLin, params_.aeroDragQuad,
            params_.yawRateDamping, params_.pitchRollDamping,
            params_.name + "_AeroDamp"));
    }
    attached_ = true;
}

void Vehicle::update(double t, double dt) {
    DriverInput in = driver_ ? driver_->update(t, dt, *this) : manualInput_;
    in.throttle = std::max(-1.0, std::min(1.0, in.throttle));
    in.brake    = std::max( 0.0, std::min(1.0, in.brake));
    in.steering = std::max(-1.0, std::min(1.0, in.steering));

    // Steering wheel angle = input × maxSteeringWheel (≈ ratio × maxRoadAngle).
    double maxSW = params_.steering.steeringRatio * params_.steering.maxRoadAngle;
    steering_->setSteeringWheelAngle(in.steering * maxSW);
    drivetrain_->update(in.throttle);
    brakes_->update(in.brake);

    // Refresh per-wheel diagnostics (Fz, kappa, alpha, spin, in-contact)
    // from the converged kinematic state. Doing this here — once per macro
    // step — keeps the readouts consistent with the rest of the vehicle
    // state. Without this they would reflect the last finite-difference
    // Jacobian probe of the implicit integrator, which perturbs body
    // velocities and produces noisy / spike-y diagnostic values.
    if (road_) {
        for (auto& c : corners_) c->wheel().sampleDiagnostics(*road_);
    }
}

double Vehicle::forwardSpeed() const {
    if (!chassis_) return 0.0;
    Vec3 fwd = chassis_->bodyToWorldDirection({1.0, 0.0, 0.0});
    return chassis_->velocity.dot(fwd);
}

double Vehicle::lateralSpeed() const {
    if (!chassis_) return 0.0;
    Vec3 lat = chassis_->bodyToWorldDirection({0.0, 0.0, 1.0});
    return chassis_->velocity.dot(lat);
}

double Vehicle::yawAngle() const {
    if (!chassis_) return 0.0;
    Vec3 fwd = chassis_->bodyToWorldDirection({1.0, 0.0, 0.0});
    return std::atan2(-fwd.z, fwd.x);
}

} // namespace mb
