/**
 * Static cantilever beam deflection test.
 * Compares FEM result with Euler-Bernoulli analytical solution.
 *
 * Key insight: Linear tet elements (constant strain) are VERY stiff in bending.
 * They suffer from "shear locking" - the element can't represent the linear
 * strain field needed for bending without also generating spurious shear strain.
 * A ratio of ~0.05-0.1 vs analytical for coarse meshes is actually expected.
 * The fix needs to be in the element formulation or mesh refinement.
 */
#include <cstdio>
#include <cmath>
#include "mb/fem/ANCFTypes.h"
#include "mb/fem/FlexibleBody.h"
#include "mb/fem/FlexibleIntegrators.h"

using namespace mb;

int main()
{
    double Lx = 1.0, Ly = 0.05, Lz = 0.05;
    double E = 70e9, nu = 0.3, rho = 7800.0;
    double g = 9.81;

    double A = Ly * Lz;
    double I = Lz * Ly * Ly * Ly / 12.0;
    double w = rho * A * g;
    double delta_analytical = w * Lx * Lx * Lx * Lx / (8.0 * E * I);

    std::printf("=== Cantilever Beam Static Deflection Test ===\n");
    std::printf("Beam: %.2f x %.4f x %.4f m\n", Lx, Ly, Lz);
    std::printf("E = %.2e Pa, nu = %.2f, rho = %.1f\n", E, nu, rho);
    std::printf("I = %.6e m^4\n", I);
    std::printf("w = %.4f N/m\n", w);
    std::printf("Analytical tip deflection: %.6f mm\n\n", delta_analytical * 1e3);

    // Test with increasing mesh refinement
    struct MeshConfig
    {
        int nx, ny, nz;
    };
    MeshConfig meshes[] = {{5, 1, 1}, {10, 1, 1}, {10, 2, 2}, {20, 2, 2}, {20, 4, 4}};

    for (auto &mc : meshes)
    {
        int nx = mc.nx, ny = mc.ny, nz = mc.nz;
        auto mesh = generateBoxTetMesh(Lx, Ly, Lz, nx, ny, nz);

        ElasticMaterialProps mat{E, nu, rho, MaterialType::NeoHookean};
        auto body = FlexibleBody::fromMesh(mesh, mat, "beam", true);
        body->gravity = Vec3(0, -g, 0);
        body->dampingAlpha = 0;
        body->fixNodesOnPlane('x', 0.0, 1e-6);

        std::printf("Mesh %dx%dx%d: nodes=%zu, elems=%zu, free_pos_DOFs=%d\n",
                    nx, ny, nz, body->nodes.size(), body->elements.size(),
                    body->numFreeDof());

        // Check gravity
        auto Qg = body->computeGravityForces();
        double totalGravForce = 0;
        for (int i = 0; i < (int)body->nodes.size(); i++)
        {
            int off = i * 12;
            totalGravForce += Qg[off + 1];
        }
        std::printf("  Gravity force (Y): %.3f N (expected: %.3f N)\n",
                    totalGravForce, -rho * Lx * Ly * Lz * g);

        // Static solve
        StaticSolveOptions opts;
        opts.maxIter = 50;
        opts.tol = 1e-4;
        opts.verbose = false;
        opts.nLoadSteps = 1;

        auto result = solveStaticEquilibrium(*body, opts);

        double maxDeflY = 0;
        for (const auto &nd : body->nodes)
        {
            if (std::abs(nd.X0[0] - Lx) < 1e-6)
            {
                double dy = -(nd.q[1] - nd.X0[1]);
                if (dy > maxDeflY)
                    maxDeflY = dy;
            }
        }

        std::printf("  converged=%s iters=%d  tip_defl=%.6f mm  ratio=%.4f\n\n",
                    result.converged ? "YES" : "NO", result.iterations,
                    maxDeflY * 1e3, maxDeflY / delta_analytical);
    }

    return 0;
}
