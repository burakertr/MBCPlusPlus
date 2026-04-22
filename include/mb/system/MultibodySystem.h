#pragma once
#include "mb/core/Body.h"
#include "mb/core/RigidBody.h"
#include "mb/core/State.h"
#include "mb/constraints/Constraint.h"
#include "mb/forces/Force.h"
#include "mb/solvers/ConstraintSolver.h"
#include "mb/integrators/TimeIntegrator.h"
#include "mb/contact/ContactManager.h"
#include "mb/math/MatrixN.h"
#include <vector>
#include <memory>
#include <functional>
#include <string>
#include <map>

namespace mb {

/**
 * Analysis result: energy/momentum of the system.
 */
struct AnalysisResult {
    double kineticEnergy = 0.0;
    double potentialEnergy = 0.0;
    double totalEnergy = 0.0;
    Vec3 linearMomentum;
    Vec3 angularMomentum;
    double constraintViolation = 0.0;
    double velocityViolation = 0.0;
};

/**
 * Simulation statistics for a time step.
 */
struct SimStats {
    int steps = 0;
    int funcEvals = 0;
    int rejectedSteps = 0;
    int contactCount = 0;
    double maxConstraintViolation = 0.0;
    double wallTime = 0.0;
};

/**
 * Central manager of a multibody system.
 * Assembles M, Cq, Q, γ; creates derivative functions;
 * integrates with constraint projection (GGL stabilization).
 */
class MultibodySystem {
public:
    MultibodySystem(const std::string& name = "MBS");
    ~MultibodySystem() = default;

    // ---- Building the system ----
    void addBody(std::shared_ptr<Body> body);
    void addConstraint(std::shared_ptr<Constraint> constraint);
    void addForce(std::shared_ptr<Force> force);

    void setGravity(const Vec3& g) { gravity_ = g; }
    Vec3 getGravity() const { return gravity_; }

    void setSolver(std::shared_ptr<ConstraintSolver> solver) { solver_ = solver; }
    void setIntegrator(std::shared_ptr<TimeIntegrator> integrator) { integrator_ = integrator; }
    void setContactManager(std::shared_ptr<ContactManager> mgr) { contactManager_ = mgr; }

    // ---- State management ----
    void initialize();
    StateVector getState() const { return state_; }
    void setState(const StateVector& state);

    // ---- Assembly ----
    /// Build system mass matrix M  (nv × nv, block diagonal)
    MatrixN assembleMassMatrix() const;

    /// Build constraint Jacobian Cq  (nc × nv)
    MatrixN assembleJacobian() const;

    /// Build generalized force vector Q  (nv)
    std::vector<double> assembleForces(double t) const;

    /// Build gamma (RHS of constraint eq with Baumgarte)  (nc)
    std::vector<double> assembleGamma() const;

    /// Total number of constraint equations
    int numConstraintEquations() const;

    // ---- Solve / Step ----
    /// Solve accelerations and multipliers at an arbitrary state st.
    /// Syncs bodies from st, assembles M/Cq/Q/γ, solves KKT.
    /// Returns full-state v-space accelerations + Lagrange multipliers.
    KKTResult solveKKTAtState(double t, StateVector& st);

    /// HHT-DAE mixed KKT: M,Q assembled at α-state; Cq,γ assembled at n+1 state.
    /// This is the correct Negrut 2007 formulation.
    KKTResult solveKKTAtHHTState(double t_alpha,
                                  StateVector& s_alpha,
                                  StateVector& s_np1);

    /// After post-projection, recompute HHT aPrev_ at the projected state
    /// and inject it via setAPrev() — preserves Newmark continuity.
    void recomputeHHTAPrev(double t);

    /// Solve for accelerations and multipliers
    SolverResult solveAccelerations(double t);

    /// Create the derivative function for ODE integrators
    DerivativeFunction createDerivativeFunction();

    /// Single time step
    StepResult step(double dt);

    /// Run simulation from t0 to tf
    SimStats simulate(double tf, double dt,
        std::function<void(double, const StateVector&)> callback = nullptr);

    // ---- Constraint projection (GGL) ----
    void projectConstraintsPosition();
    void projectConstraintsVelocity();

    // ---- Analysis ----
    AnalysisResult analyze() const;

    // ---- Accessors ----
    const std::vector<std::shared_ptr<Body>>& getBodies() const { return bodies_; }
    const std::vector<std::shared_ptr<Constraint>>& getConstraints() const { return constraints_; }
    const std::vector<std::shared_ptr<Force>>& getForces() const { return forces_; }

    double getTime() const { return state_.time; }
    std::string getName() const { return name_; }

    Body* getBodyById(int id) const;
    Body* getBodyByName(const std::string& name) const;

private:
    std::string name_;
    Vec3 gravity_{0, -9.81, 0};
    double time_ = 0.0;

    std::vector<std::shared_ptr<Body>> bodies_;
    std::vector<std::shared_ptr<Constraint>> constraints_;
    std::vector<std::shared_ptr<Force>> forces_;

    std::shared_ptr<ConstraintSolver> solver_;
    std::shared_ptr<TimeIntegrator> integrator_;
    std::shared_ptr<ContactManager> contactManager_;

    StateVector state_;
    bool initialized_ = false;

    // Dynamic-only body tracking (mirrors TS dynamicBodyList / dynamicVOffsets)
    std::vector<int> dynBodyIndices_;   // indices into bodies_
    std::vector<int> dynVOffsets_;      // velocity offsets in dynamic-only space
    int totalDynNv_ = 0;               // total dynamic DOFs
    std::map<int, int> dynBodyIdToIdx_; // body.id → index in dynBodyIndices_

    void rebuildDynamicIndices();

    // Helper: body pointers (raw) for State operations
    std::vector<Body*> bodyPtrs() const;

    // Copy state into body members
    void syncStateToBodie() const;
    // Copy body members into state
    void syncBodiestoState();
};

} // namespace mb
