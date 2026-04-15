#include "mb/constraints/Constraint.h"
#include <cmath>

namespace mb {

Constraint::Constraint(const std::string& name) : name(name) {}

void Constraint::setBaumgarteParameters(double alpha, double beta) {
    alpha_ = alpha;
    beta_ = beta;
}

MatrixN Constraint::getJacobian() const {
    auto jac = computeJacobian();
    auto bodyIds = getBodyIds();
    int nc = numEquations();
    int numBds = static_cast<int>(bodyIds.size());
    
    MatrixN J(nc, numBds * 6);
    for (int eq = 0; eq < nc; eq++) {
        for (int j = 0; j < 6; j++) {
            J.set(eq, j, jac.J1.get(eq, j));
            if (numBds > 1) {
                J.set(eq, 6 + j, jac.J2.get(eq, j));
            }
        }
    }
    return J;
}

ConstraintViolation Constraint::getViolation() const {
    return computeViolation();
}

std::vector<double> Constraint::computeVelocityViolation() const {
    auto jac = computeJacobian();
    auto bodyIds = getBodyIds();
    int nc = numEquations();
    
    std::vector<double> Cdot(nc, 0.0);
    
    // For each body, get velocity and compute Cq * v
    // This is a simplified version - proper implementation needs body references
    return Cdot;
}

std::vector<double> Constraint::getGamma() const {
    int nc = numEquations();
    auto violation = computeViolation();
    auto convective = computeConvectiveTerm();
    
    std::vector<double> gamma(nc);
    auto velViol = computeVelocityViolation();
    
    for (int i = 0; i < nc; i++) {
        // γ = -convective - 2α·Ċ - β²·C
        double C = violation.position[i];
        double Cdot = (i < static_cast<int>(velViol.size())) ? velViol[i] : 0.0;
        gamma[i] = -convective[i] - 2.0 * alpha_ * Cdot - beta_ * beta_ * C;
    }
    return gamma;
}

std::vector<double> Constraint::computeConvectiveTerm() const {
    // Numerical convective term using finite differences
    // Default: zero (subclasses can override with analytical version)
    return std::vector<double>(numEquations(), 0.0);
}

} // namespace mb
