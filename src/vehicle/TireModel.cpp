#include "mb/vehicle/TireModel.h"
#include <algorithm>
#include <cmath>

namespace mb {

TireForces LinearTireModel::evaluate(const TireSlip& s) const {
    TireForces f;
    // Sign convention: kappa = (Vroll - Vx)/V, so kappa>0 means wheel
    // spinning faster than ground (traction). The longitudinal friction
    // force on the wheel is then in the +fwd direction (drives chassis).
    // Likewise alpha = atan(Vy/|Vx|) so alpha>0 means contact slipping in
    // +lat; the lateral force opposes that, i.e. -lat direction.
    f.Fx =  Cs_ * s.kappa;
    f.Fy = -Calpha_ * s.alpha;
    const double Fmax = mu_ * std::max(s.Fz, 0.0);
    const double mag  = std::sqrt(f.Fx * f.Fx + f.Fy * f.Fy);
    if (mag > Fmax && mag > 1e-9) {
        const double k = Fmax / mag;
        f.Fx *= k;
        f.Fy *= k;
    }
    return f;
}

namespace {
inline double magicFormula(double x, double B, double C, double D, double E) {
    const double Bx = B * x;
    return D * std::sin(C * std::atan(Bx - E * (Bx - std::atan(Bx))));
}
} // namespace

TireForces PacejkaTireModel::evaluate(const TireSlip& s) const {
    TireForces f;
    const double Fz = std::max(s.Fz, 0.0);
    if (Fz <= 0.0) return f;

    const double Dx = p_.Dx * p_.mu * Fz;
    const double Dy = p_.Dy * p_.mu * Fz;

    double Fx0 = magicFormula(s.kappa, p_.Bx, p_.Cx, Dx, p_.Ex);
    double Fy0 = magicFormula(s.alpha, p_.By, p_.Cy, Dy, p_.Ey);

    // Note: lateral slip α convention here -> Fy opposes α.
    Fy0 = -Fy0;

    // Combined slip via friction-ellipse projection.
    const double Fmax = p_.mu * Fz;
    const double mag  = std::sqrt(Fx0 * Fx0 + Fy0 * Fy0);
    if (mag > Fmax && mag > 1e-9) {
        const double k = Fmax / mag;
        Fx0 *= k;
        Fy0 *= k;
    }

    f.Fx = Fx0;
    f.Fy = Fy0;

    // Self-aligning moment: Mz = -t(α) * Fy.
    double trail = p_.trail0 * (1.0 - std::min(1.0, std::abs(s.alpha) / std::max(p_.alphaPeak, 1e-6)));
    if (trail < 0.0) trail = 0.0;
    f.Mz = -trail * Fy0;

    return f;
}

} // namespace mb
