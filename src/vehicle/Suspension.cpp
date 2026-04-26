#include "mb/vehicle/Suspension.h"
#include "mb/system/MultibodySystem.h"
#include "mb/vehicle/Road.h"
#include <algorithm>
#include <cmath>

namespace mb {

// ---------------- SuspensionCorner ----------------

SuspensionCorner::SuspensionCorner(const std::string& name,
                                   const SuspensionCornerParams& params,
                                   std::shared_ptr<RigidBody> chassis,
                                   std::shared_ptr<TireModel> tireModel)
    : name_(name), params_(params), chassis_(std::move(chassis)) {
    // Carrier and knuckle: small inertias, modeled as boxes for visualisation.
    carrier_ = RigidBody::createBox(params.carrierMass, 0.06, 0.10, 0.10,
                                    name + "_Carrier");
    knuckle_ = RigidBody::createBox(params.knuckleMass, 0.08, 0.12, 0.12,
                                    name + "_Knuckle");
    wheel_ = std::make_unique<Wheel>(name + "_Wheel", params.wheel,
                                     std::move(tireModel));
}

void SuspensionCorner::attachToSystem(MultibodySystem& sys, const Road& road) {
    // ----- Initial poses -----
    // Strategy: place the entire unsprung mass (carrier + knuckle + wheel) so
    // that the wheel just touches the ground (no penetration), then compute
    // the spring's effective rest length so it produces exactly designLoad of
    // upward force on the chassis at t = 0. This puts the system extremely
    // close to static equilibrium and avoids the violent transient that would
    // otherwise occur with a misplaced or pre-stretched spring.
    Vec3 attachW = chassis_->bodyToWorld(params_.chassisAttachLocal);

    // Lateral outward direction in world (from chassis frame Z, signed).
    double sideSign = params_.isLeftSide ? -1.0 : 1.0;
    Vec3 outwardLocal{0.0, 0.0, sideSign};
    Vec3 outwardWorld = chassis_->orientation.rotate(outwardLocal);

    // Wheel centre at attach (x,z) and one radius above the ground (use the
    // ground height at attach (x,z) — for a flat road this is exact; for a
    // heightmap it's an excellent first approximation).
    // Place the wheel pre-compressed so the tire spring already supports the
    // sprung+unsprung weight at t = 0 (eliminates the violent settling
    // transient that would otherwise saturate the implicit step).
    double yGround = road.height(attachW.x, attachW.z);
    double unsprung = params_.carrierMass + params_.knuckleMass + params_.wheel.mass;
    double tirePen  = 0.0;
    if (params_.wheel.kz > 0.0)
        tirePen = (params_.designLoad + unsprung * 9.81) / params_.wheel.kz;
    // Cap to a fraction of radius — unphysical otherwise.
    tirePen = std::min(tirePen, 0.5 * params_.wheel.radius);
    double wheelY  = yGround + params_.wheel.radius - tirePen;

    // Carrier and knuckle: at chassis-attach (x,z), at wheel-centre height,
    // oriented like the chassis. (Steer joint and spin attach offsets are
    // expressed in their own frames.)
    carrier_->position = Vec3{attachW.x, wheelY, attachW.z};
    carrier_->orientation = chassis_->orientation;
    knuckle_->position = carrier_->position;
    knuckle_->orientation = carrier_->orientation;

    // Wheel: centred on the knuckle (zero scrub). Placing the wheel exactly
    // on the kingpin axis means longitudinal tyre force Fx (traction, brake
    // or rolling resistance) produces NO moment about the steering axis,
    // so the corner does not self-steer toe-in under load.
    Vec3 wheelCentre = knuckle_->position;
    wheel_->body()->position = wheelCentre;
    // Spin-axis convention: orient ALL wheels so that the wheel's local +Y
    // (the cylinder's symmetry axis, which carries the polar inertia) points
    // in the chassis -Z direction (vehicle LEFT) regardless of which side the
    // corner is on. This makes the world spin axis the same for every wheel,
    // which in turn makes "positive lastSpin == rolling forward" a uniform
    // convention and removes the per-side sign flip that would otherwise
    // garble drive/brake torque application.
    Vec3 yAxisLocal{0.0, 1.0, 0.0};
    Vec3 spinDirLocal{0.0, 0.0, -1.0};
    Quaternion wheelOri = Quaternion::fromVectors(yAxisLocal, spinDirLocal);
    wheel_->body()->orientation = chassis_->orientation.multiply(wheelOri);

    // ----- Spring rest length: pre-compress to balance designLoad -----
    // Initial spring length L0 = |attachW - carrier_origin| (purely vertical).
    Vec3 springTopLocal    = params_.chassisAttachLocal;
    Vec3 springBottomLocal = Vec3(0.0, 0.0, 0.0);
    double L0 = attachW.sub(carrier_->position).length();
    double k  = std::max(1.0, params_.springDamper.stiffness);
    double preCompress = (params_.designLoad > 0.0) ? (params_.designLoad / k) : 0.0;
    double effectiveRest = L0 + preCompress;

    // ----- Joints -----
    // 1) Carrier ↔ Chassis: prismatic along chassis local Y. The constraint's
    // local point on the carrier is the carrier origin (0,0,0): the perp
    // components of (chassis_attach - carrier_origin) must vanish, which is
    // exactly satisfied at startup since both share (x,z).
    travelJoint_ = std::make_shared<PrismaticJoint>(
        chassis_.get(), carrier_.get(),
        params_.chassisAttachLocal,
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        name_ + "_Travel");

    // 2) Knuckle ↔ Carrier:
    //    * Steerable (front): revolute about carrier local Y (kingpin),
    //      driven by the SteeringActuator PD controller.
    //    * Non-steerable (rear): FixedJoint that removes all 6 relative
    //      DOF, so tyre Mz, drivetrain reaction torques and chassis roll
    //      cannot leak into a free kingpin DOF and self-steer the rear
    //      axle. Cleaner than relying on a stiff PD lock at finite kp.
    if (params_.steerable) {
        steerJoint_ = std::make_shared<RevoluteJoint>(
            carrier_.get(), knuckle_.get(),
            Vec3(0.0, 0.0, 0.0),
            Vec3(0.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            // Light viscous damping. The PD steering actuator does the
            // heavy lifting; high joint damping alone would couple poorly
            // with implicit integrators.
            5.0,
            name_ + "_Steer");
    } else {
        steerJoint_ = std::make_shared<FixedJoint>(
            carrier_.get(), knuckle_.get(),
            Vec3(0.0, 0.0, 0.0),
            Vec3(0.0, 0.0, 0.0),
            name_ + "_SteerLock");
    }

    // 3) Wheel ↔ Knuckle: revolute about the unified wheel spin axis
    // (chassis -Z in world, which is wheel local +Y after the orientation
    // applied above). The knuckle local axis is the same chassis -Z.
    // Attach point: knuckle origin (zero scrub — wheel is centred on
    // the kingpin axis to avoid Fx-induced self-steer).
    spinJoint_ = std::make_shared<RevoluteJoint>(
        knuckle_.get(), wheel_->body().get(),
        Vec3(0.0, 0.0, 0.0),                       // attach point on knuckle
        Vec3(0.0, 0.0, 0.0),                       // wheel centre
        spinDirLocal,                              // knuckle local axis (-Z)
        Vec3(0.0, 1.0, 0.0),                       // wheel local axis (+Y)
        0.0,
        name_ + "_Spin");

    // Wire the knuckle pointer to the wheel so TireForceElement can apply
    // drive / brake / Mz reaction torques back to the chassis through the
    // knuckle (Newton's 3rd law) and project them onto the knuckle's spin
    // axis (avoids self-steer from the wheel's instantaneous-axis drift).
    wheel_->setKnuckle(knuckle_.get());

    // ----- Spring-damper between chassis and carrier (with preload) -----
    spring_ = std::make_shared<SpringDamper>(
        chassis_.get(), carrier_.get(),
        springTopLocal, springBottomLocal,
        params_.springDamper.stiffness,
        params_.springDamper.damping,
        effectiveRest,
        name_ + "_Spring");

    // ----- Add to system -----
    sys.addBody(carrier_);
    sys.addBody(knuckle_);
    sys.addBody(wheel_->body());

    sys.addConstraint(travelJoint_);
    sys.addConstraint(steerJoint_);
    sys.addConstraint(spinJoint_);

    sys.addForce(spring_);

    // Tire force element.
    sys.addForce(std::make_shared<TireForceElement>(wheel_.get(), &road,
                                                    name_ + "_Tire"));
}

double SuspensionCorner::currentSteerAngle() const {
    if (!carrier_ || !knuckle_) return 0.0;
    // Measure angle between knuckle's forward axis and carrier's forward axis,
    // projected on carrier-Y plane. Forward axis: chassis local +X.
    Vec3 carrierFwd = carrier_->bodyToWorldDirection({1.0, 0.0, 0.0});
    Vec3 knuckleFwd = knuckle_->bodyToWorldDirection({1.0, 0.0, 0.0});
    Vec3 carrierUp  = carrier_->bodyToWorldDirection({0.0, 1.0, 0.0});
    // Project both onto plane perpendicular to carrierUp.
    auto projPlane = [&](const Vec3& v) {
        return v.sub(carrierUp.scale(v.dot(carrierUp))).normalize();
    };
    Vec3 a = projPlane(carrierFwd);
    Vec3 b = projPlane(knuckleFwd);
    double c = std::max(-1.0, std::min(1.0, a.dot(b)));
    double sgn = (a.cross(b).dot(carrierUp) >= 0.0) ? 1.0 : -1.0;
    return sgn * std::acos(c);
}

double SuspensionCorner::currentCompression() const {
    if (!chassis_ || !carrier_) return 0.0;
    Vec3 attachW = chassis_->bodyToWorld(params_.chassisAttachLocal);
    Vec3 topW    = carrier_->bodyToWorld({0.0, 0.0, 0.0});
    double L = attachW.sub(topW).length();
    return params_.springDamper.restLength - L;
}

// ---------------- SteeringActuator ----------------

SteeringActuator::SteeringActuator(SuspensionCorner* corner,
                                   double kp, double kd, double maxTorque,
                                   const std::string& name)
    : Force(name), corner_(corner), kp_(kp), kd_(kd), maxTorque_(maxTorque) {}

void SteeringActuator::apply(double /*t*/) {
    // Active on every corner. For non-steerable rear corners the Steering
    // controller leaves steerTarget_ at 0, so this becomes a stiff PD lock
    // that keeps the knuckle aligned with the carrier — much more robust to
    // tyre Mz and drivetrain reaction perturbations than relying on viscous
    // joint damping alone (the latter cannot prevent quasi-static drift).
    if (!corner_) return;
    double theta = corner_->currentSteerAngle();
    double target = corner_->steerTarget();
    // Relative angular velocity about kingpin (carrier-Y world).
    Vec3 axisW = corner_->carrier()->bodyToWorldDirection({0.0, 1.0, 0.0}).normalize();
    double omegaRel = corner_->knuckle()->angularVelocity.sub(
                          corner_->carrier()->angularVelocity).dot(axisW);
    double tau = kp_ * (target - theta) - kd_ * omegaRel;
    if (tau >  maxTorque_) tau =  maxTorque_;
    if (tau < -maxTorque_) tau = -maxTorque_;
    corner_->knuckle()->applyTorque(axisW.scale(tau));
    corner_->carrier()->applyTorque(axisW.scale(-tau));
}

// ---------------- AntiRollBar ----------------

AntiRollBar::AntiRollBar(SuspensionCorner* left, SuspensionCorner* right,
                         double stiffness, const std::string& name)
    : Force(name), left_(left), right_(right), k_(stiffness) {}

void AntiRollBar::apply(double /*t*/) {
    if (!left_ || !right_) return;
    double zL = left_->currentCompression();
    double zR = right_->currentCompression();
    double dz = zL - zR;
    // Force magnitude opposing roll: push up the more-compressed side, down the other.
    double F = k_ * dz;
    Vec3 up = left_->carrier()->bodyToWorldDirection({0.0, 1.0, 0.0}).normalize();
    Vec3 fL = up.scale(-F); // if zL > zR (left lower) push left UP -> -F? compression positive means lower; push UP = +up
    // Re-derive: compression positive = carrier moved closer to chassis = chassis side dropped.
    // Roll-resisting force should push the more-compressed carrier DOWNWARD relative to chassis
    // (extending its spring back) — i.e. push carrier down. So force on more-compressed carrier = -up*|F|.
    fL = up.scale(-F);          // F>0 (left more compressed) => push left carrier down
    Vec3 fR = up.scale(+F);     // and right carrier up
    left_->carrier()->applyForce(fL);
    right_->carrier()->applyForce(fR);
    // Reaction on chassis: equal & opposite at attachment points.
    Vec3 attachL = left_->params().chassisAttachLocal;
    Vec3 attachR = right_->params().chassisAttachLocal;
    auto* chassis = left_->chassis().get();
    chassis->applyForceAtPoint(fL.negate(), chassis->bodyToWorld(attachL));
    chassis->applyForceAtPoint(fR.negate(), chassis->bodyToWorld(attachR));
}

} // namespace mb
