#pragma once
#include "mb/core/State.h"
#include "mb/core/Body.h"
#include <functional>
#include <vector>

namespace mb {

// Derivative function: f(t, state) → d(state)/dt
using DerivativeFunction = std::function<StateVector(double, StateVector&)>;

struct IntegratorConfig {
    double minStep = 1e-8;
    double maxStep = 0.01;
    double absTol = 1e-6;
    double relTol = 1e-6;
    bool adaptive = false;
};

struct StepResult {
    StateVector state;
    double dt;        // Actual step size used
    double nextDt;    // Suggested next step size
    bool accepted;
};

struct IntegrationResult {
    StateVector state;
    int steps;
    int evaluations;
    int rejected;
};

/**
 * Abstract base class for time integrators
 */
class TimeIntegrator {
public:
    TimeIntegrator(const IntegratorConfig& config = {}) : config_(config) {}
    virtual ~TimeIntegrator() = default;

    virtual StepResult step(double t, StateVector& state, double dt,
                            DerivativeFunction& f) = 0;

    IntegrationResult integrate(
        StateVector& state, double t0, double tf, double dt,
        DerivativeFunction& f,
        std::function<void(double, StateVector&)> callback = nullptr
    );

    IntegratorConfig getConfig() const { return config_; }

    // Called after external state modification (e.g. constraint projection)
    virtual void invalidateCache() {}

    /**
     * If false, the integrator handles velocity-level constraints internally
     * (e.g. HHT-α uses consistent velocity update) and the outer
     * MultibodySystem::step() should skip the separate velocity projection
     * to avoid double-counting which bleeds energy.
     */
    virtual bool needsVelocityProjection() const { return true; }

protected:
    IntegratorConfig config_;
};

} // namespace mb
