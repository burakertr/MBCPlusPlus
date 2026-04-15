#pragma once
#include "TimeIntegrator.h"

namespace mb {

/**
 * Classic 4th-order Runge-Kutta
 */
class RungeKutta4 : public TimeIntegrator {
public:
    RungeKutta4(const IntegratorConfig& config = {}) : TimeIntegrator(config) {}
    StepResult step(double t, StateVector& state, double dt,
                    DerivativeFunction& f) override;
};

/**
 * Runge-Kutta-Fehlberg 4(5) with adaptive step size
 */
class RungeKuttaFehlberg45 : public TimeIntegrator {
public:
    RungeKuttaFehlberg45(const IntegratorConfig& config = {});
    StepResult step(double t, StateVector& state, double dt,
                    DerivativeFunction& f) override;
};

/**
 * Dormand-Prince 4(5) with FSAL optimization
 */
class DormandPrince45 : public TimeIntegrator {
public:
    DormandPrince45(const IntegratorConfig& config = {});
    StepResult step(double t, StateVector& state, double dt,
                    DerivativeFunction& f) override;
    void invalidateCache() override;

private:
    bool hasCachedK7_ = false;
    StateVector cachedK7_;
};

} // namespace mb
