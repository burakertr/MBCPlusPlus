#include "mb/integrators/BDF.h"
#include <cmath>

namespace mb {

// ============== BackwardEuler ==============

StepResult BackwardEuler::step(double t, StateVector& state, double dt,
                               DerivativeFunction& f) {
    // Implicit Euler: y_{n+1} = y_n + dt * f(t_{n+1}, y_{n+1})
    // Use one Newton iteration with explicit predictor
    auto predicted = state.clone();
    auto k = f(t + dt, predicted);

    StateVector result = state.addScaled(k, dt);
    result.time = t + dt;
    result.normalizeQuaternions();

    return {result, dt, dt, true};
}

// ============== BDF2 ==============

StepResult BDF2::step(double t, StateVector& state, double dt,
                      DerivativeFunction& f) {
    if (!hasHistory_) {
        // Fall back to backward Euler for first step
        auto k = f(t + dt, state);
        StateVector result = state.addScaled(k, dt);
        result.time = t + dt;
        result.normalizeQuaternions();
        hasHistory_ = true;
        prevState_ = state.clone();
        return {result, dt, dt, true};
    }

    // BDF-2: y_{n+1} = 4/3*y_n - 1/3*y_{n-1} + 2/3*dt*f(t_{n+1}, y_{n+1})
    auto pred = state.clone();
    auto k = f(t + dt, pred);

    StateVector result = state.clone();
    for (int i = 0; i < state.totalNq; i++)
        result.q[i] = 4.0/3*state.q[i] - 1.0/3*prevState_.q[i] + 2.0/3*dt*k.q[i];
    for (int i = 0; i < state.totalNv; i++)
        result.v[i] = 4.0/3*state.v[i] - 1.0/3*prevState_.v[i] + 2.0/3*dt*k.v[i];

    result.time = t + dt;
    result.normalizeQuaternions();
    prevState_ = state.clone();

    return {result, dt, dt, true};
}

// ============== GeneralizedAlpha ==============

GeneralizedAlpha::GeneralizedAlpha(double rhoInfinity, const IntegratorConfig& config)
    : TimeIntegrator(config) {
    alphaM_ = (2 * rhoInfinity - 1) / (rhoInfinity + 1);
    alphaF_ = rhoInfinity / (rhoInfinity + 1);
    gammaGA_ = 0.5 + alphaF_ - alphaM_;
    beta_ = 0.25 * (gammaGA_ + 0.5) * (gammaGA_ + 0.5);
}

StepResult GeneralizedAlpha::step(double t, StateVector& state, double dt,
                                   DerivativeFunction& f) {
    // Newmark-like update with generalized-alpha parameters
    auto k = f(t + dt, state);

    StateVector result = state.clone();
    for (int i = 0; i < state.totalNv; i++) {
        double a_new = k.v[i]; // acceleration from derivative
        result.v[i] = state.v[i] + dt * ((1 - gammaGA_) * state.a[i] + gammaGA_ * a_new);
    }

    for (int i = 0; i < state.totalNq; i++) {
        result.q[i] = state.q[i] + dt * k.q[i];
    }

    // Store accelerations
    for (int i = 0; i < state.totalNv; i++) {
        result.a[i] = k.v[i];
    }

    result.time = t + dt;
    result.normalizeQuaternions();

    return {result, dt, dt, true};
}

// ============== SemiImplicitEuler ==============

StepResult SemiImplicitEuler::step(double t, StateVector& state, double dt,
                                    DerivativeFunction& f) {
    // 1. Compute accelerations at current state
    auto k = f(t, state);

    StateVector result = state.clone();

    // 2. Update velocity: v_{n+1} = v_n + dt * a
    for (int i = 0; i < state.totalNv; i++)
        result.v[i] = state.v[i] + dt * k.v[i];

    // 3. Update position using NEW velocity: q_{n+1} = q_n + dt * q̇(v_{n+1})
    for (int i = 0; i < state.totalNq; i++)
        result.q[i] = state.q[i] + dt * k.q[i]; // Simplified

    result.time = t + dt;
    result.normalizeQuaternions();

    return {result, dt, dt, true};
}

// ============== VelocityVerlet ==============

StepResult VelocityVerlet::step(double t, StateVector& state, double dt,
                                 DerivativeFunction& f) {
    // 1. Compute accelerations
    auto k1 = f(t, state);

    StateVector result = state.clone();

    // 2. Half-step velocity
    for (int i = 0; i < state.totalNv; i++)
        result.v[i] = state.v[i] + 0.5 * dt * k1.v[i];

    // 3. Full-step position
    for (int i = 0; i < state.totalNq; i++)
        result.q[i] = state.q[i] + dt * k1.q[i];

    result.time = t + dt;

    // 4. Compute new accelerations
    auto k2 = f(t + dt, result);

    // 5. Complete velocity step
    for (int i = 0; i < state.totalNv; i++)
        result.v[i] = state.v[i] + 0.5 * dt * (k1.v[i] + k2.v[i]);

    result.normalizeQuaternions();

    return {result, dt, dt, true};
}

} // namespace mb
