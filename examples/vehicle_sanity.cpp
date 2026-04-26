// Quick self-test: build a sedan on a flat road and run a few steps.
// Verifies attach + step + basic sanity (Fz balance, no NaN).
#include "mb/system/MultibodySystem.h"
#include "mb/solvers/NewtonRaphson.h"
#include "mb/integrators/BDF.h"
#include "mb/vehicle/Vehicle.h"
#include "mb/vehicle/Road.h"
#include <cstdio>
#include <cmath>

using namespace mb;

int main() {
    MultibodySystem sys("vehicle_sanity");
    sys.setGravity({0.0, -9.81, 0.0});
    sys.addBody(RigidBody::createGround("Ground"));

    FlatRoad road{0.0, 1.0};
    VehicleParams vp = Vehicle::sedanDefaults("Sedan");
    vp.initialPosition = {0.0, 0.7, 0.0};
    vp.antiRollFront = 0.0;
    vp.antiRollRear  = 0.0;
    Vehicle veh(vp);
    veh.attachToSystem(sys, road);

    SolverConfig sc; sc.maxIterations = 50; sc.tolerance = 1e-7; sc.warmStart = true;
    sys.setSolver(std::make_shared<NewtonRaphsonSolver>(sc));
    IntegratorConfig ic; ic.adaptive = false; ic.relTol = 1e-4; ic.absTol = 1e-6;
    sys.setIntegrator(std::make_shared<BDF2>(ic));
    sys.initialize();

    const double dt = 1e-4;
    for (int k = 0; k < 5000; ++k) { // 0.5 s settle
        veh.update(sys.getTime(), dt);
        sys.step(dt);
        if (k % 100 == 0) {
            printf("k=%d t=%.4f y_chassis=%.4f vy=%.4f Fz=[",
                   k, sys.getTime(), veh.chassis()->position.y,
                   veh.chassis()->velocity.y);
            for (int i = 0; i < 4; ++i) printf(" %.0f", veh.wheel(i).lastFz());
            printf("]  pen=[");
            for (int i = 0; i < 4; ++i) {
                double y = veh.wheel(i).body()->position.y;
                printf(" %.4f", y);
            }
            printf("]\n");
        }
    }
    double Fz = 0.0;
    for (int i = 0; i < 4; ++i) Fz += veh.wheel(i).lastFz();
    double mg = vp.chassisMass * 9.81;
    // sprung+unsprung (ignore unsprung for headline check)
    printf("After settle: chassis y = %.3f m, ΣFz = %.0f N, m·g (sprung) = %.0f N\n",
           veh.chassis()->position.y, Fz, mg);

    // Apply throttle.
    veh.setManualInput({0.5, 0.0, 0.0});
    for (int k = 0; k < 5000; ++k) {
        veh.update(sys.getTime(), dt);
        sys.step(dt);
        if (k % 250 == 0) {
            Vec3 omegaC = veh.chassis()->angularVelocity;
            printf("acc k=%d t=%.3f x=%.3f vx=%.3f vy=%.4f omegaC=(%.3f,%.3f,%.3f) wheelSpin=[",
                   k, sys.getTime(), veh.chassis()->position.x,
                   veh.forwardSpeed(), veh.chassis()->velocity.y,
                   omegaC.x, omegaC.y, omegaC.z);
            for (int i = 0; i < 4; ++i) printf(" %.2f", veh.wheel(i).lastSpin());
            printf("] kappa=[");
            for (int i = 0; i < 4; ++i) printf(" %.3f", veh.wheel(i).lastKappa());
            printf("]\n");
            if (!std::isfinite(veh.forwardSpeed())) { printf("STOP NaN\n"); break; }
        }
    }
    printf("After accel: v = %.2f m/s (%.1f km/h), x = %.2f m\n",
           veh.forwardSpeed(), veh.forwardSpeed() * 3.6,
           veh.chassis()->position.x);

    if (!std::isfinite(veh.forwardSpeed())) {
        fprintf(stderr, "NaN in forward speed\n");
        return 1;
    }

    // ── Stress test mirroring vehicle_qt: full throttle for 5 s, no
    //    steering input, watch yaw drift / per-wheel steer angle. Also
    //    reports per-corner steerAngle to catch any toe-in build-up.
    printf("\n[STRESS] full throttle 1.0, 5 s, watching for yaw drift...\n");
    veh.setManualInput({1.0, 0.0, 0.0});
    double yaw0 = veh.yawAngle();
    for (int k = 0; k < 50000; ++k) {
        veh.update(sys.getTime(), dt);
        sys.step(dt);
        if (k % 1000 == 0) {
            Vec3 om = veh.chassis()->angularVelocity;
            printf("str k=%5d t=%.2f vx=%6.2f yaw=%+.4f wy=%+.4f steer=[",
                   k, sys.getTime(), veh.forwardSpeed(),
                   veh.yawAngle() - yaw0, om.y);
            for (int i = 0; i < 4; ++i) {
                double a = veh.corner(i).currentSteerAngle();
                printf(" %+.4f", a);
            }
            printf("] kap=[");
            for (int i = 0; i < 4; ++i) printf(" %+.3f", veh.wheel(i).lastKappa());
            printf("] alp=[");
            for (int i = 0; i < 4; ++i) printf(" %+.3f", veh.wheel(i).lastAlpha());
            printf("]\n");
            if (!std::isfinite(veh.forwardSpeed())) { printf("STRESS NaN\n"); return 1; }
        }
    }
    printf("[STRESS] final v=%.2f m/s, yaw drift=%.4f rad (%.1f deg)\n",
           veh.forwardSpeed(), veh.yawAngle() - yaw0,
           (veh.yawAngle() - yaw0) * 180.0 / 3.14159265);

    return 0;
}
