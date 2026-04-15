#pragma once
#include "Body.h"
#include <vector>
#include <memory>

namespace mb {

/**
 * Combined system state for ODE integration.
 * Stores q (positions), v (velocities), a (accelerations), lambda (multipliers)
 * with per-body offset tables supporting variable DOF.
 */
class StateVector {
public:
    double time = 0.0;

    std::vector<double> q;  // Generalized positions
    std::vector<double> v;  // Generalized velocities
    std::vector<double> a;  // Accelerations
    std::vector<double> lambda; // Lagrange multipliers

    // Per-body offsets
    std::vector<int> qOffsets;
    std::vector<int> vOffsets;
    std::vector<int> nqPerBody;
    std::vector<int> nvPerBody;

    int numBodies = 0;
    int numConstraints = 0;
    int totalNq = 0;
    int totalNv = 0;

    StateVector() = default;

    static StateVector fromBodies(const std::vector<Body*>& bodies, int numConstraints = 0);

    int dof() const { return totalNv; }
    int constrainedDof() const { return totalNv - numConstraints; }

    StateVector clone() const;
    void copyFrom(const StateVector& other);

    // Add scaled: this += scale * other (for integrator operations)
    StateVector addScaled(const StateVector& other, double scale) const;

    // Normalize quaternions
    void normalizeQuaternions();

    // Compute q̇ from velocities
    std::vector<double> computeQDot(const std::vector<Body*>& bodies) const;

    // Copy state to/from body
    void copyToBody(int bodyIndex, Body* body) const;
    void copyFromBody(int bodyIndex, Body* body);

    // Accessors by body index
    Vec3 getPosition(int bodyIndex) const;
    Vec3 getVelocity(int bodyIndex) const;
    Quaternion getOrientation(int bodyIndex) const;
    Vec3 getAngularVelocity(int bodyIndex) const;
};

} // namespace mb
