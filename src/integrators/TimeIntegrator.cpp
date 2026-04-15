#include "mb/integrators/TimeIntegrator.h"
#include <algorithm>
#include <cmath>

namespace mb {

IntegrationResult TimeIntegrator::integrate(
    StateVector& state, double t0, double tf, double dt,
    DerivativeFunction& f,
    std::function<void(double, StateVector&)> callback
) {
    double t = t0;
    double currentDt = std::min(dt, config_.maxStep);
    int steps = 0, evals = 0, rejected = 0;

    StateVector current = state.clone();

    while (t < tf - 1e-14) {
        currentDt = std::min(currentDt, tf - t);
        auto result = step(t, current, currentDt, f);

        if (result.accepted) {
            current = result.state;
            t = current.time;
            currentDt = std::min(result.nextDt, config_.maxStep);
            steps++;

            if (callback) callback(t, current);
        } else {
            currentDt = std::min(result.nextDt, config_.maxStep);
            rejected++;
        }
        evals++;
    }

    state = current;
    return {current, steps, evals, rejected};
}

} // namespace mb
