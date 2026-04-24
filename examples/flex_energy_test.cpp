/**
 * Headless energy-conservation test for the ANCF cantilever beam.
 * Prints Total Energy at each frame to verify conservation.
 */
#include <cstdio>
#include <cmath>
#include "mb/fem/ANCFTypes.h"
#include "mb/fem/FlexibleBody.h"
#include "mb/fem/FlexibleIntegrators.h"
#include "mb/fem/MeshGenerators.h"

using namespace mb;

int main() {
    auto mesh = generateBoxTetMesh(1.0, 0.05, 0.05, 10, 2, 2);
    ElasticMaterialProps mat{2e7, 0.3, 7800.0, MaterialType::StVenantKirchhoff};
    auto body = FlexibleBody::fromMesh(mesh, mat, "beam", true);
    body->gravity = Vec3(0, -9.81, 0);
    body->dampingAlpha = 0;
    body->fixNodesOnPlane('x', 0.0, 1e-6);

    FlexDOPRI45 dopri(*body);
    dopri.absTol  = 1e-8;
    dopri.relTol  = 1e-6;
    dopri.maxStep = 0.005;
    dopri.dtCurrent = 1e-4;

    std::printf("%8s %14s %14s %14s %14s %14s %14s\n",
                "time", "KE", "StrainE", "GravPE", "TotalE", "dE/E0", "maxZ_disp");

    double E0 = 0;
    for (int frame = 0; frame <= 100; frame++) {
        auto r = dopri.step(0.01);
        double t = (frame + 1) * 0.01;

        double SE = r.strainEnergy;
        double KE = r.kineticEnergy;
        double GP = r.gravitationalPE;
        double TE = SE + KE + GP;

        if (frame == 0) E0 = TE;
        double drift = (E0 != 0) ? (TE - E0) / std::abs(E0) : 0;

        // Measure max Z displacement
        double maxZdisp = 0;
        for (const auto& nd : body->nodes) {
            double dz = std::abs(nd.q[2] - nd.X0[2]);
            if (dz > maxZdisp) maxZdisp = dz;
        }

        std::printf("%8.3f %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e\n",
                    t, KE, SE, GP, TE, drift, maxZdisp);
    }
    std::printf("\nSteps: %d  Rejects: %d\n", dopri.totalSteps, dopri.totalRejects);
    return 0;
}
