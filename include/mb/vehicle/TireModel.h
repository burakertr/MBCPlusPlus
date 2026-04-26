#pragma once
#include <memory>
#include <string>

namespace mb {

/**
 * Tire kinematic state at a contact patch.
 *  kappa : longitudinal slip ratio = (omega*R - Vx) / max(|Vx|, eps)
 *  alpha : slip angle [rad]        = atan2(Vy, |Vx|)  (signed)
 *  Fz    : vertical load [N]       (>= 0)
 *  gamma : camber angle [rad]      (positive = top tilts outward)
 *  Vx    : longitudinal speed of the contact patch in tire frame [m/s]
 */
struct TireSlip {
    double kappa = 0.0;
    double alpha = 0.0;
    double Fz    = 0.0;
    double gamma = 0.0;
    double Vx    = 0.0;
};

/**
 * Tire forces in the tire (contact-patch) frame.
 *  Fx : longitudinal force [N]   (positive = drive)
 *  Fy : lateral force      [N]   (positive = +y of tire frame)
 *  Mz : self-aligning moment about the contact normal [N·m]
 */
struct TireForces {
    double Fx = 0.0;
    double Fy = 0.0;
    double Mz = 0.0;
};

/**
 * Abstract tire model: maps slip state to tyre-frame forces.
 */
class TireModel {
public:
    virtual ~TireModel() = default;
    virtual TireForces evaluate(const TireSlip& s) const = 0;
    virtual std::string name() const { return "TireModel"; }
};

/**
 * Linear slip model (debug / sanity reference):
 *   Fx = -Cs * kappa,   Fy = -Calpha * alpha
 * Both clamped by friction ellipse |F| <= mu * Fz. No Mz.
 */
class LinearTireModel : public TireModel {
public:
    LinearTireModel(double cornering_stiffness = 80000.0,
                    double longitudinal_stiffness = 100000.0,
                    double mu = 1.0)
        : Calpha_(cornering_stiffness), Cs_(longitudinal_stiffness), mu_(mu) {}

    TireForces evaluate(const TireSlip& s) const override;
    std::string name() const override { return "LinearTireModel"; }

private:
    double Calpha_;
    double Cs_;
    double mu_;
};

/**
 * Pacejka Magic Formula (simplified, MF-94-style) tyre.
 * Pure-slip Fx and Fy; combined slip via friction-ellipse projection.
 *   F = D * sin( C * atan( B*x - E*(B*x - atan(B*x)) ) )
 * with x = kappa for Fx, x = alpha for Fy. D scales linearly with Fz.
 * Self-aligning moment is approximated as Mz = -t * Fy with pneumatic
 * trail t = t0 * (1 - |alpha|/alpha_peak) clamped to >= 0.
 */
struct PacejkaParams {
    // Longitudinal
    double Bx = 10.0;   // stiffness
    double Cx = 1.65;   // shape
    double Dx = 1.0;    // peak factor (multiplied by mu * Fz)
    double Ex = 0.97;   // curvature
    // Lateral
    double By = 8.5;
    double Cy = 1.30;
    double Dy = 1.0;
    double Ey = -1.0;
    // Friction limit
    double mu = 1.0;
    // Self-aligning trail
    double trail0      = 0.05;   // [m] at zero slip
    double alphaPeak   = 0.20;   // [rad] slip where trail collapses
};

class PacejkaTireModel : public TireModel {
public:
    explicit PacejkaTireModel(const PacejkaParams& p = {}) : p_(p) {}

    TireForces evaluate(const TireSlip& s) const override;
    std::string name() const override { return "PacejkaTireModel"; }

    const PacejkaParams& params() const { return p_; }
    void setParams(const PacejkaParams& p) { p_ = p; }

private:
    PacejkaParams p_;
};

} // namespace mb
