#include "mb/vehicle/Wheel.h"
#include <algorithm>
#include <cmath>

namespace mb {

Wheel::Wheel(const std::string& name,
             const WheelParams& params,
             std::shared_ptr<TireModel> tireModel)
    : name_(name), params_(params), tireModel_(std::move(tireModel)) {
    body_ = RigidBody::createCylinder(params.mass, params.radius, params.width, name);
}

// ---------------- TireForceElement ----------------

TireForceElement::TireForceElement(Wheel* wheel, const Road* road,
                                   const std::string& name)
    : Force(name), wheel_(wheel), road_(road) {}

void TireForceElement::apply(double /*t*/) {
    if (!wheel_ || !road_) return;
    wheel_->evaluateContact_(*road_, /*applyForces=*/true);
}

// ---------------- Wheel::sampleDiagnostics ----------------

void Wheel::sampleDiagnostics(const Road& road) {
    evaluateContact_(road, /*applyForces=*/false);
}

// ---------------- Wheel::evaluateContact_ ----------------

void Wheel::evaluateContact_(const Road& road, bool applyForces) {
    auto* body = body_.get();
    if (!body) return;

    const auto& p = params_;

    // ----- Geometry: spin axis & contact patch -----
    // Wheel local +Y is the spin axis (RigidBody::createCylinder convention).
    Vec3 spinAxisW = body->bodyToWorldDirection({0.0, 1.0, 0.0}).normalize();

    // Drivetrain spin axis: defined by the KNUCKLE, not the wheel. Drive,
    // brake and self-aligning reaction torques are applied along this axis.
    // Using the wheel's own instantaneous axis is wrong because (i) tiny
    // numerical drift in wheel orientation would project drive torque onto
    // the kingpin DOF and self-steer the corner, and (ii) without a chassis
    // reaction the system violates Newton's 3rd law and the chassis yaws.
    // The knuckle's local axis matching the wheel's spin axis (chassis -Z
    // after the orientation set in Suspension::attachToSystem) is (0,0,-1).
    auto* knuckle = knuckle_;
    Vec3 axisDriveW = knuckle
        ? knuckle->bodyToWorldDirection({0.0, 0.0, -1.0}).normalize()
        : spinAxisW;

    // Surface normal at the wheel-centre's planar projection.
    Vec3 nW = road.normal(body->position.x, body->position.z).normalize();

    // The contact-patch direction is perpendicular to the spin axis, lying in
    // the plane of the surface normal: d = normalize( n × spin × spin )... use
    // the standard formulation: project -n onto plane perpendicular to spin.
    Vec3 down = nW.negate();
    Vec3 toContact = down.sub(spinAxisW.scale(spinAxisW.dot(down))).normalize();
    if (toContact.lengthSquared() < 1e-12) {
        // Spin axis nearly parallel to surface normal: no contact possible.
        lastInContact_ = false;
        lastFz_ = 0.0;
        return;
    }
    Vec3 contactPt = body->position.add(toContact.scale(p.radius));

    // Penetration: ground elevation at the contact-patch (x,z) compared to
    // the contact point's height along the surface normal.
    double yGround = road.height(contactPt.x, contactPt.z);
    Vec3 ground{contactPt.x, yGround, contactPt.z};
    double signedDist = contactPt.sub(ground).dot(nW); // >0: above surface
    double pen = -signedDist;                          // >0: penetrating

    if (pen <= 0.0) {
        lastInContact_ = false;
        lastFz_ = 0.0;
        // Brake/drive torques still applied to the spinning wheel even airborne.
        if (applyForces && driveTorque_ != 0.0) {
            Vec3 tau = axisDriveW.scale(driveTorque_);
            body->applyTorque(tau);
            if (knuckle) knuckle->applyTorque(tau.negate());
        }
        if (applyForces && brakeTorque_ > 0.0) {
            double omegaSpin = body->angularVelocity.dot(axisDriveW);
            double tauMag = -std::copysign(brakeTorque_, omegaSpin);
            Vec3 tau = axisDriveW.scale(tauMag);
            body->applyTorque(tau);
            if (knuckle) knuckle->applyTorque(tau.negate());
        }
        return;
    }

    // ----- Velocity at contact patch (for tangential slip-velocity) -----
    Vec3 vCp = body->getPointVelocity(contactPt);
    // Penetration rate (positive while compressing).
    double penRate = -vCp.dot(nW);

    // Vertical normal force.
    double Fz = p.kz * pen + p.cz * penRate;
    if (Fz < 0.0) Fz = 0.0;

    // ----- Tire frame at the contact patch -----
    // Wheel forward direction = spin × n (right-hand rule with Y-up vehicle).
    Vec3 fwd = spinAxisW.cross(nW);
    double fwdLen = fwd.length();
    if (fwdLen < 1e-9) {
        // Degenerate (axis aligned with surface normal): no tractive force.
        lastInContact_ = true;
        lastFz_ = Fz;
        if (applyForces) body->applyForceAtPoint(nW.scale(Fz), contactPt);
        return;
    }
    fwd = fwd.scale(1.0 / fwdLen);
    Vec3 lat = nW.cross(fwd).normalize();   // tyre lateral axis in surface plane

    // Slip ratio uses the WHEEL CENTRE forward velocity (not the contact-point
    // velocity — the latter already subtracts ω·R, which would make pure
    // rolling read as κ = 1 instead of 0). Slip angle similarly uses centre.
    Vec3 vCenter = body->velocity;
    double Vx = vCenter.dot(fwd);
    double Vy = vCenter.dot(lat);

    // Wheel spin rate about its axis.
    double omegaSpin = body->angularVelocity.dot(spinAxisW);
    // Roll velocity (forward speed equivalent to current spin rate).
    double Vroll = omegaSpin * p.radius;

    // Slip ratio (longitudinal). Use a velocity guard at low speed so the
    // ratio stays bounded near zero forward speed.
    double Vguard = std::max(std::abs(Vx), p.vEps);
    double kappa  = (Vroll - Vx) / Vguard;

    // Slip angle (lateral). Conventional sign: alpha = atan2(Vy, |Vx|).
    double alpha = std::atan2(Vy, Vguard);

    // Camber: angle between spin axis and surface plane.
    double sinGamma = std::max(-1.0, std::min(1.0, spinAxisW.dot(nW)));
    double gamma = std::asin(sinGamma);

    // ----- Tyre model -----
    TireSlip slip{kappa, alpha, Fz, gamma, Vx};
    TireForces tf = tireModel_->evaluate(slip);

    // Rolling resistance (always opposes forward velocity).
    double Frr = p.rolling_resistance * Fz;
    if (std::abs(Vx) > 1e-3) {
        tf.Fx -= std::copysign(Frr, Vx);
    }

    // ----- Apply forces and torques -----
    if (applyForces) {
        Vec3 Fworld = nW.scale(Fz).add(fwd.scale(tf.Fx)).add(lat.scale(tf.Fy));
        body->applyForceAtPoint(Fworld, contactPt);

        // Self-aligning moment about the contact normal. Apply equal-and-opposite
        // reaction to the knuckle so the moment is reacted by the steering chain
        // (otherwise it leaks into the wheel's free spin DOF and slowly steers
        // the corner).
        if (tf.Mz != 0.0) {
            Vec3 tauMz = nW.scale(tf.Mz);
            body->applyTorque(tauMz);
            if (knuckle) knuckle->applyTorque(tauMz.negate());
        }

        // Drive torque about knuckle's spin axis (drivetrain reaction reaches
        // the chassis via the knuckle → carrier → chassis joint chain).
        if (driveTorque_ != 0.0) {
            Vec3 tau = axisDriveW.scale(driveTorque_);
            body->applyTorque(tau);
            if (knuckle) knuckle->applyTorque(tau.negate());
        }
        // Brake torque opposes spin direction (also reacted on knuckle).
        if (brakeTorque_ > 0.0) {
            double tauMag = -std::copysign(brakeTorque_, omegaSpin);
            Vec3 tau = axisDriveW.scale(tauMag);
            body->applyTorque(tau);
            if (knuckle) knuckle->applyTorque(tau.negate());
        }
    }

    lastInContact_ = true;
    lastFz_    = Fz;
    lastKappa_ = kappa;
    lastAlpha_ = alpha;
    lastSpin_  = omegaSpin;
}

} // namespace mb
