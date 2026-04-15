#pragma once
#include "mb/math/Vec3.h"
#include "mb/math/MatrixN.h"
#include "mb/core/Body.h"
#include <vector>
#include <string>
#include <memory>

namespace mb {

struct ConstraintViolation {
    std::vector<double> position;
    std::vector<double> velocity;
};

struct JacobianResult {
    MatrixN J1; // Jacobian w.r.t. body 1 (nc × 6)
    MatrixN J2; // Jacobian w.r.t. body 2 (nc × 6)
};

/**
 * Abstract base class for constraints.
 * Uses Baumgarte stabilization: γ = -2αĊ - β²C - convective
 */
class Constraint {
public:
    std::string name;
    double time = 0.0;

    Constraint(const std::string& name = "Constraint");
    virtual ~Constraint() = default;

    virtual int numEquations() const = 0;
    virtual std::vector<int> getBodyIds() const = 0;

    // Core constraint interface
    virtual ConstraintViolation computeViolation() const = 0;
    virtual JacobianResult computeJacobian() const = 0;

    // Compute velocity-level violation: Ċ = Cq * v
    virtual std::vector<double> computeVelocityViolation() const;

    // Get assembled Jacobian (nc × (6 * numBodies)) 
    MatrixN getJacobian() const;

    // Get violation (position)
    ConstraintViolation getViolation() const;

    // Get gamma (RHS with Baumgarte stabilization)
    std::vector<double> getGamma() const;

    // Baumgarte parameters
    void setBaumgarteParameters(double alpha, double beta);

protected:
    double alpha_ = 5.0;
    double beta_ = 5.0;

    // Numerical convective term (finite differences on Cq·q̇)
    virtual std::vector<double> computeConvectiveTerm() const;
};

} // namespace mb
