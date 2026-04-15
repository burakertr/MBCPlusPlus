/**
 * Double-pendulum example using the MBC++ multibody library.
 * Demonstrates: bodies, spherical joints, gravity, integration, analysis.
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

    std::cout << "=== MBC++ Double Pendulum Example ===" << std::endl;

    // Create system
    MultibodySystem sys("DoublePendulum");
    sys.setGravity(Vec3(0, -9.81, 0));

    // Ground (static)
    auto ground = RigidBody::createGround("Ground");
    sys.addBody(ground);

    // Pendulum link 1: 1 kg rod, 1 m long
    auto link1 = RigidBody::createRod(1.0, 1.0, 0.02, "Link1");
    link1->position = Vec3(0.5, 0, 0); // Horizontal to the right
    link1->orientation = Quaternion(0.7071067811865476, 0, 0, -0.7071067811865476); // 90° Z rotation
    sys.addBody(link1);

    // Pendulum link 2: 0.5 kg rod, 0.8 m long
    auto link2 = RigidBody::createRod(0.5, 0.8, 0.02, "Link2");
    link2->position = Vec3(1.4, 0, 0); // Horizontal, beyond link1
    link2->orientation = Quaternion(0.7071067811865476, 0, 0, -0.7071067811865476);
    sys.addBody(link2);

    // Joint 1: ground → link1 at origin
    auto joint1 = std::make_shared<SphericalJoint>(
        ground.get(), link1.get(),
        Vec3(0, 0, 0),     // anchor on ground
        Vec3(0, -0.5, 0),  // negative-Y end of link1
        "Joint1"
    );
    sys.addConstraint(joint1);

    // Joint 2: link1 → link2
    auto joint2 = std::make_shared<SphericalJoint>(
        link1.get(), link2.get(),
        Vec3(0, 0.5, 0),   // positive-Y end of link1
        Vec3(0, -0.4, 0),  // negative-Y end of link2 (closest to link1)
        "Joint2"
    );
    sys.addConstraint(joint2);

    // Gravity handled by sys.setGravity() via assembleForces()

    // Set solver and integrator (match TS: DOPRI45 adaptive, tight tolerances)
    sys.setSolver(std::make_shared<DirectSolver>());
    IntegratorConfig cfg;
    cfg.absTol = 1e-10;
    cfg.relTol = 1e-8;
    cfg.adaptive = true;
    cfg.maxStep = 0.001;
    cfg.minStep = 1e-12;
    sys.setIntegrator(std::make_shared<DormandPrince45>(cfg));

    // No initial angular velocity — let gravity pull the horizontal pendulum down.
    // (Setting angular velocity without consistent linear velocities for all
    //  connected bodies creates velocity constraint violations that projection
    //  must fix, which injects/removes energy.)
    // link1->angularVelocity = Vec3(0, 0, 2.0);

    // Initialize
    sys.initialize();

    // Print header
    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::setw(8) << "Time"
              << std::setw(12) << "KE"
              << std::setw(12) << "PE"
              << std::setw(12) << "Total E"
              << std::setw(12) << "Constr."
              << std::setw(14) << "Link1 Y"
              << std::setw(14) << "Link2 Y"
              << std::endl;
    std::cout << std::string(82, '-') << std::endl;

    // Simulate 2 seconds
    double dt = 0.001;
    double tEnd = 2.0;
    int printInterval = 100; // Print every 100 steps
    int stepCount = 0;

    auto stats = sys.simulate(tEnd, dt,
        [&](double t, const StateVector& st) {
            stepCount++;
            if (stepCount % printInterval == 0) {
                auto analysis = sys.analyze();
                std::cout << std::setw(8) << t
                          << std::setw(12) << analysis.kineticEnergy
                          << std::setw(12) << analysis.potentialEnergy
                          << std::setw(12) << analysis.totalEnergy
                          << std::setw(12) << analysis.constraintViolation
                          << std::setw(14) << link1->position.y
                          << std::setw(14) << link2->position.y
                          << std::endl;
            }
        }
    );

    // Final report
    auto final_analysis = sys.analyze();
    std::cout << std::string(82, '-') << std::endl;
    std::cout << "Simulation complete." << std::endl;
    std::cout << "  Steps:         " << stats.steps << std::endl;
    std::cout << "  Func evals:    " << stats.funcEvals << std::endl;
    std::cout << "  Rejected:      " << stats.rejectedSteps << std::endl;
    std::cout << "  Wall time:     " << std::setprecision(3) << stats.wallTime << " s" << std::endl;
    std::cout << "  Final energy:  " << std::setprecision(6) << final_analysis.totalEnergy << std::endl;
    std::cout << "  Max constr:    " << final_analysis.constraintViolation << std::endl;
    std::cout << "  Link1 pos:     (" << link1->position.x << ", "
              << link1->position.y << ", " << link1->position.z << ")" << std::endl;
    std::cout << "  Link2 pos:     (" << link2->position.x << ", "
              << link2->position.y << ", " << link2->position.z << ")" << std::endl;

    return 0;
}
