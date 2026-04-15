#include "mb/integrators/RungeKutta.h"
#include <cmath>
#include <algorithm>

namespace mb {

// ============== RungeKutta4 ==============

StepResult RungeKutta4::step(double t, StateVector& state, double dt,
                             DerivativeFunction& f) {
    // k1
    auto k1 = f(t, state);

    // k2 — normalize quaternions at each intermediate stage (matches TS)
    auto s2 = state.addScaled(k1, dt / 2.0);
    s2.normalizeQuaternions();
    auto k2 = f(t + dt / 2.0, s2);

    // k3
    auto s3 = state.addScaled(k2, dt / 2.0);
    s3.normalizeQuaternions();
    auto k3 = f(t + dt / 2.0, s3);

    // k4
    auto s4 = state.addScaled(k3, dt);
    s4.normalizeQuaternions();
    auto k4 = f(t + dt, s4);

    // Combine: y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    StateVector result = state.clone();
    for (int i = 0; i < state.totalNq; i++) {
        result.q[i] = state.q[i] + dt / 6.0 * (k1.q[i] + 2*k2.q[i] + 2*k3.q[i] + k4.q[i]);
    }
    for (int i = 0; i < state.totalNv; i++) {
        result.v[i] = state.v[i] + dt / 6.0 * (k1.v[i] + 2*k2.v[i] + 2*k3.v[i] + k4.v[i]);
    }

    result.time = t + dt;
    result.normalizeQuaternions();

    return {result, dt, dt, true};
}

// ============== RungeKuttaFehlberg45 ==============

RungeKuttaFehlberg45::RungeKuttaFehlberg45(const IntegratorConfig& config)
    : TimeIntegrator(config) {
    config_.adaptive = true;
}

StepResult RungeKuttaFehlberg45::step(double t, StateVector& state, double dt,
                                       DerivativeFunction& f) {
    // RKF45 Butcher tableau coefficients
    auto k1 = f(t, state);

    auto s2 = state.addScaled(k1, dt / 4.0);
    s2.normalizeQuaternions();
    auto k2 = f(t + dt / 4.0, s2);

    // s3 = state + dt*(3/32*k1 + 9/32*k2)
    auto s3 = state.clone();
    for (int i = 0; i < state.totalNq; i++)
        s3.q[i] = state.q[i] + dt * (3.0/32*k1.q[i] + 9.0/32*k2.q[i]);
    for (int i = 0; i < state.totalNv; i++)
        s3.v[i] = state.v[i] + dt * (3.0/32*k1.v[i] + 9.0/32*k2.v[i]);
    s3.normalizeQuaternions();
    auto k3 = f(t + 3.0*dt/8.0, s3);

    // s4
    auto s4 = state.clone();
    for (int i = 0; i < state.totalNq; i++)
        s4.q[i] = state.q[i] + dt * (1932.0/2197*k1.q[i] - 7200.0/2197*k2.q[i] + 7296.0/2197*k3.q[i]);
    for (int i = 0; i < state.totalNv; i++)
        s4.v[i] = state.v[i] + dt * (1932.0/2197*k1.v[i] - 7200.0/2197*k2.v[i] + 7296.0/2197*k3.v[i]);
    s4.normalizeQuaternions();
    auto k4 = f(t + 12.0*dt/13.0, s4);

    // s5
    auto s5 = state.clone();
    for (int i = 0; i < state.totalNq; i++)
        s5.q[i] = state.q[i] + dt * (439.0/216*k1.q[i] - 8*k2.q[i] + 3680.0/513*k3.q[i] - 845.0/4104*k4.q[i]);
    for (int i = 0; i < state.totalNv; i++)
        s5.v[i] = state.v[i] + dt * (439.0/216*k1.v[i] - 8*k2.v[i] + 3680.0/513*k3.v[i] - 845.0/4104*k4.v[i]);
    s5.normalizeQuaternions();
    auto k5 = f(t + dt, s5);

    // s6
    auto s6 = state.clone();
    for (int i = 0; i < state.totalNq; i++)
        s6.q[i] = state.q[i] + dt * (-8.0/27*k1.q[i] + 2*k2.q[i] - 3544.0/2565*k3.q[i] + 1859.0/4104*k4.q[i] - 11.0/40*k5.q[i]);
    for (int i = 0; i < state.totalNv; i++)
        s6.v[i] = state.v[i] + dt * (-8.0/27*k1.v[i] + 2*k2.v[i] - 3544.0/2565*k3.v[i] + 1859.0/4104*k4.v[i] - 11.0/40*k5.v[i]);
    s6.normalizeQuaternions();
    auto k6 = f(t + dt / 2.0, s6);

    // 4th order solution
    StateVector y4 = state.clone();
    for (int i = 0; i < state.totalNq; i++)
        y4.q[i] = state.q[i] + dt * (25.0/216*k1.q[i] + 1408.0/2565*k3.q[i] + 2197.0/4104*k4.q[i] - 1.0/5*k5.q[i]);
    for (int i = 0; i < state.totalNv; i++)
        y4.v[i] = state.v[i] + dt * (25.0/216*k1.v[i] + 1408.0/2565*k3.v[i] + 2197.0/4104*k4.v[i] - 1.0/5*k5.v[i]);

    // 5th order solution
    StateVector y5 = state.clone();
    for (int i = 0; i < state.totalNq; i++)
        y5.q[i] = state.q[i] + dt * (16.0/135*k1.q[i] + 6656.0/12825*k3.q[i] + 28561.0/56430*k4.q[i] - 9.0/50*k5.q[i] + 2.0/55*k6.q[i]);
    for (int i = 0; i < state.totalNv; i++)
        y5.v[i] = state.v[i] + dt * (16.0/135*k1.v[i] + 6656.0/12825*k3.v[i] + 28561.0/56430*k4.v[i] - 9.0/50*k5.v[i] + 2.0/55*k6.v[i]);

    // Error estimate
    double err = 0;
    for (int i = 0; i < state.totalNv; i++) {
        double e = std::abs(y5.v[i] - y4.v[i]);
        double scale = config_.absTol + config_.relTol * std::abs(y5.v[i]);
        err = std::max(err, e / scale);
    }

    double safety = 0.9;
    double nextDt;

    if (err <= 1.0) {
        y5.time = t + dt;
        y5.normalizeQuaternions();
        nextDt = (err > 1e-10) ? safety * dt * std::pow(1.0 / err, 0.2) : dt * 2.0;
        nextDt = std::max(config_.minStep, std::min(nextDt, config_.maxStep));
        return {y5, dt, nextDt, true};
    } else {
        nextDt = safety * dt * std::pow(1.0 / err, 0.25);
        nextDt = std::max(config_.minStep, nextDt);
        return {state, dt, nextDt, false};
    }
}

// ============== DormandPrince45 ==============

DormandPrince45::DormandPrince45(const IntegratorConfig& config)
    : TimeIntegrator(config) {
    config_.adaptive = true;
}

void DormandPrince45::invalidateCache() {
    hasCachedK7_ = false;
}

StepResult DormandPrince45::step(double t, StateVector& state, double dt,
                                  DerivativeFunction& f) {
    // DOPRI5 coefficients
    constexpr double a21 = 1.0/5;
    constexpr double a31 = 3.0/40, a32 = 9.0/40;
    constexpr double a41 = 44.0/45, a42 = -56.0/15, a43 = 32.0/9;
    constexpr double a51 = 19372.0/6561, a52 = -25360.0/2187, a53 = 64448.0/6561, a54 = -212.0/729;
    constexpr double a61 = 9017.0/3168, a62 = -355.0/33, a63 = 46732.0/5247, a64 = 49.0/176, a65 = -5103.0/18656;
    constexpr double a71 = 35.0/384, a73 = 500.0/1113, a74 = 125.0/192, a75 = -2187.0/6784, a76 = 11.0/84;

    // Error coefficients
    constexpr double e1 = 71.0/57600, e3 = -71.0/16695, e4 = 71.0/1920, e5 = -17253.0/339200, e6 = 22.0/525, e7 = -1.0/40;

    StateVector k1 = hasCachedK7_ ? cachedK7_ : f(t, state);

    auto s2 = state.addScaled(k1, dt * a21);
    s2.normalizeQuaternions();
    auto k2 = f(t + dt/5, s2);

    auto s3 = state.clone();
    for (int i = 0; i < state.totalNq; i++)
        s3.q[i] = state.q[i] + dt*(a31*k1.q[i] + a32*k2.q[i]);
    for (int i = 0; i < state.totalNv; i++)
        s3.v[i] = state.v[i] + dt*(a31*k1.v[i] + a32*k2.v[i]);
    s3.normalizeQuaternions();
    auto k3 = f(t + 3*dt/10, s3);

    auto s4 = state.clone();
    for (int i = 0; i < state.totalNq; i++)
        s4.q[i] = state.q[i] + dt*(a41*k1.q[i] + a42*k2.q[i] + a43*k3.q[i]);
    for (int i = 0; i < state.totalNv; i++)
        s4.v[i] = state.v[i] + dt*(a41*k1.v[i] + a42*k2.v[i] + a43*k3.v[i]);
    s4.normalizeQuaternions();
    auto k4 = f(t + 4*dt/5, s4);

    auto s5 = state.clone();
    for (int i = 0; i < state.totalNq; i++)
        s5.q[i] = state.q[i] + dt*(a51*k1.q[i] + a52*k2.q[i] + a53*k3.q[i] + a54*k4.q[i]);
    for (int i = 0; i < state.totalNv; i++)
        s5.v[i] = state.v[i] + dt*(a51*k1.v[i] + a52*k2.v[i] + a53*k3.v[i] + a54*k4.v[i]);
    s5.normalizeQuaternions();
    auto k5 = f(t + 8*dt/9, s5);

    auto s6 = state.clone();
    for (int i = 0; i < state.totalNq; i++)
        s6.q[i] = state.q[i] + dt*(a61*k1.q[i] + a62*k2.q[i] + a63*k3.q[i] + a64*k4.q[i] + a65*k5.q[i]);
    for (int i = 0; i < state.totalNv; i++)
        s6.v[i] = state.v[i] + dt*(a61*k1.v[i] + a62*k2.v[i] + a63*k3.v[i] + a64*k4.v[i] + a65*k5.v[i]);
    s6.normalizeQuaternions();
    auto k6 = f(t + dt, s6);

    // 5th order solution (used as the result)
    StateVector y5 = state.clone();
    for (int i = 0; i < state.totalNq; i++)
        y5.q[i] = state.q[i] + dt*(a71*k1.q[i] + a73*k3.q[i] + a74*k4.q[i] + a75*k5.q[i] + a76*k6.q[i]);
    for (int i = 0; i < state.totalNv; i++)
        y5.v[i] = state.v[i] + dt*(a71*k1.v[i] + a73*k3.v[i] + a74*k4.v[i] + a75*k5.v[i] + a76*k6.v[i]);
    y5.time = t + dt;

    // k7 for FSAL and error estimate
    auto k7 = f(t + dt, y5);

    // Error estimate (check both v and q for proper adaptive control)
    double err = 0;
    for (int i = 0; i < state.totalNv; i++) {
        double ei = dt * (e1*k1.v[i] + e3*k3.v[i] + e4*k4.v[i] + e5*k5.v[i] + e6*k6.v[i] + e7*k7.v[i]);
        double scale = config_.absTol + config_.relTol * std::max(std::abs(state.v[i]), std::abs(y5.v[i]));
        err = std::max(err, std::abs(ei) / scale);
    }
    for (int i = 0; i < state.totalNq; i++) {
        double ei = dt * (e1*k1.q[i] + e3*k3.q[i] + e4*k4.q[i] + e5*k5.q[i] + e6*k6.q[i] + e7*k7.q[i]);
        double scale = config_.absTol + config_.relTol * std::max(std::abs(state.q[i]), std::abs(y5.q[i]));
        err = std::max(err, std::abs(ei) / scale);
    }

    double safety = 0.9;
    double nextDt;

    if (err <= 1.0) {
        y5.normalizeQuaternions();
        hasCachedK7_ = true;
        cachedK7_ = k7;
        nextDt = (err > 1e-10) ? safety * dt * std::pow(1.0 / err, 0.2) : dt * 2.0;
        nextDt = std::max(config_.minStep, std::min(nextDt, config_.maxStep));
        return {y5, dt, nextDt, true};
    } else {
        hasCachedK7_ = false;
        nextDt = safety * dt * std::pow(1.0 / err, 0.25);
        nextDt = std::max(config_.minStep, nextDt);
        return {state, dt, nextDt, false};
    }
}

} // namespace mb
