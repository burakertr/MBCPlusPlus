/**
 * Energy conservation test - single pendulum.
 * Simplest possible system to isolate drift cause.
 */
#include "mb/core/RigidBody.h"
#include "mb/constraints/SphericalJoint.h"
#include "mb/solvers/DirectSolver.h"
#include "mb/integrators/RungeKutta.h"
#include "mb/system/MultibodySystem.h"
#include <iostream>
#include <iomanip>
#include <cmath>

int main() {
    using namespace mb;
    std::cout << "=== Single Pendulum Energy Test ===" << std::endl;

    // Create system
    MultibodySystem sys("SinglePendulum");
    sys.setGravity(Vec3(0, -9.81, 0));

    // Ground
    auto ground = RigidBody::createGround("Ground");
    sys.addBody(ground);

    // Rod: 1 kg, 1 m long, starts horizontal
    auto rod = RigidBody::createRod(1.0, 1.0, 0.02, "Rod");
    rod->position = Vec3(0.5, 0, 0);
    // Rotate 90° around Z: w=cos(45°), z=sin(45°)  
    rod->orientation = Quaternion(0.7071067811865476, 0, 0, 0.7071067811865476);
    sys.addBody(rod);

    // Joint at origin, rod's local (0, 0.5, 0) = top end
    auto joint = std::make_shared<SphericalJoint>(
        rod.get(), ground.get(),
        Vec3(0, 0.5, 0),  // top of rod in local frame
        Vec3(0, 0, 0),     // origin on ground
        "Pivot"
    );
    sys.addConstraint(joint);

    // Setup solver/integrator
    sys.setSolver(std::make_shared<DirectSolver>());

    // Test 1: RK4 fixed step
    {
        sys.setIntegrator(std::make_shared<RungeKutta4>());
        sys.initialize();

        auto a0 = sys.analyze();
        double E0 = a0.totalEnergy;
        std::cout << "\n--- RK4, dt=0.001 ---" << std::endl;
        std::cout << "Initial E = " << std::setprecision(10) << E0 << std::endl;
        std::cout << "Initial constraint = " << a0.constraintViolation << std::endl;

        double dt = 0.001;
        for (int i = 0; i < 2000; i++) {
            sys.step(dt);
        }
        auto a1 = sys.analyze();
        std::cout << "t=2.0  E = " << a1.totalEnergy 
                  << "  drift = " << (a1.totalEnergy - E0)
                  << "  rel = " << std::abs(a1.totalEnergy - E0) / std::max(1e-14, std::abs(E0))
                  << "  constr = " << a1.constraintViolation
                  << std::endl;
    }

    // Rebuild for test 2: DOPRI45
    {
        MultibodySystem sys2("SinglePendulum2");
        sys2.setGravity(Vec3(0, -9.81, 0));

        auto g2 = RigidBody::createGround("Ground");
        sys2.addBody(g2);

        auto r2 = RigidBody::createRod(1.0, 1.0, 0.02, "Rod");
        r2->position = Vec3(0.5, 0, 0);
        r2->orientation = Quaternion(0.7071067811865476, 0, 0, 0.7071067811865476);
        sys2.addBody(r2);

        auto j2 = std::make_shared<SphericalJoint>(
            r2.get(), g2.get(),
            Vec3(0, 0.5, 0), Vec3(0, 0, 0), "Pivot");
        sys2.addConstraint(j2);

        sys2.setSolver(std::make_shared<DirectSolver>());
        IntegratorConfig cfg;
        cfg.absTol = 1e-10;
        cfg.relTol = 1e-8;
        cfg.adaptive = true;
        cfg.maxStep = 0.001;
        cfg.minStep = 1e-12;
        sys2.setIntegrator(std::make_shared<DormandPrince45>(cfg));
        sys2.initialize();

        auto a0 = sys2.analyze();
        double E0 = a0.totalEnergy;
        std::cout << "\n--- DOPRI45 adaptive, maxStep=0.001 ---" << std::endl;
        std::cout << "Initial E = " << std::setprecision(10) << E0 << std::endl;

        double dt = 0.001;
        for (int i = 0; i < 2000; i++) {
            sys2.step(dt);
        }
        auto a1 = sys2.analyze();
        std::cout << "t=2.0  E = " << a1.totalEnergy 
                  << "  drift = " << (a1.totalEnergy - E0)
                  << "  rel = " << std::abs(a1.totalEnergy - E0) / std::max(1e-14, std::abs(E0))
                  << "  constr = " << a1.constraintViolation
                  << std::endl;
    }

    // Test 3: RK4 WITHOUT projection (pure Baumgarte)
    {
        MultibodySystem sys3("SinglePendulum3");
        sys3.setGravity(Vec3(0, -9.81, 0));

        auto g3 = RigidBody::createGround("Ground");
        sys3.addBody(g3);

        auto r3 = RigidBody::createRod(1.0, 1.0, 0.02, "Rod");
        r3->position = Vec3(0.5, 0, 0);
        r3->orientation = Quaternion(0.7071067811865476, 0, 0, 0.7071067811865476);
        sys3.addBody(r3);

        auto j3 = std::make_shared<SphericalJoint>(
            r3.get(), g3.get(),
            Vec3(0, 0.5, 0), Vec3(0, 0, 0), "Pivot");
        sys3.addConstraint(j3);

        sys3.setSolver(std::make_shared<DirectSolver>());
        sys3.setIntegrator(std::make_shared<RungeKutta4>());
        sys3.initialize();

        auto a0 = sys3.analyze();
        double E0 = a0.totalEnergy;
        std::cout << "\n--- RK4 NO PROJECTION (pure Baumgarte) ---" << std::endl;
        std::cout << "Initial E = " << std::setprecision(10) << E0 << std::endl;

        double dt = 0.001;
        // Override to call integrate directly (no projection)
        auto derivFunc = sys3.createDerivativeFunction();
        auto integ = std::make_shared<RungeKutta4>();

        StateVector st = sys3.getState();
        for (int i = 0; i < 2000; i++) {
            auto result = integ->step(st.time, st, dt, derivFunc);
            st = result.state;
            st.normalizeQuaternions();
        }
        sys3.setState(st);

        auto a1 = sys3.analyze();
        std::cout << "t=2.0  E = " << a1.totalEnergy 
                  << "  drift = " << (a1.totalEnergy - E0)
                  << "  constr = " << a1.constraintViolation
                  << std::endl;
    }

    return 0;
}
