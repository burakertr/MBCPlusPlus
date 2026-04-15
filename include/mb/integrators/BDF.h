#pragma once
#include "TimeIntegrator.h"

namespace mb {

/**
 * Backward Euler (BDF-1) implicit integrator
 */
class BackwardEuler : public TimeIntegrator {
public:
    BackwardEuler(const IntegratorConfig& config = {}) : TimeIntegrator(config) {}
    StepResult step(double t, StateVector& state, double dt,
                    DerivativeFunction& f) override;
};

/**
 * BDF-2 (2nd order implicit)
 */
class BDF2 : public TimeIntegrator {
public:
    BDF2(const IntegratorConfig& config = {}) : TimeIntegrator(config) {}
    StepResult step(double t, StateVector& state, double dt,
                    DerivativeFunction& f) override;

private:
    bool hasHistory_ = false;
    StateVector prevState_;
};

/**
 * Generalized-α integrator for structural dynamics
 */
class GeneralizedAlpha : public TimeIntegrator {
public:
    GeneralizedAlpha(double rhoInfinity = 0.9, const IntegratorConfig& config = {});
    StepResult step(double t, StateVector& state, double dt,
                    DerivativeFunction& f) override;

private:
    double alphaM_, alphaF_, beta_, gammaGA_;
};

/**
 * Semi-implicit Euler (symplectic)
 */
class SemiImplicitEuler : public TimeIntegrator {
public:
    SemiImplicitEuler(const IntegratorConfig& config = {}) : TimeIntegrator(config) {}
    StepResult step(double t, StateVector& state, double dt,
                    DerivativeFunction& f) override;
};

/**
 * Velocity Verlet (symplectic, 2nd order)
 */
class VelocityVerlet : public TimeIntegrator {
public:
    VelocityVerlet(const IntegratorConfig& config = {}) : TimeIntegrator(config) {}
    StepResult step(double t, StateVector& state, double dt,
                    DerivativeFunction& f) override;
};

} // namespace mb
