/**
 * pendulum_benchmark.cpp
 *
 * Double-pendulum benchmark — GUI yok, saf hesaplama.
 * Aynı sistem (pendulum_qt.cpp ile birebir), 30 saniye simüle edilir.
 *
 * Çıktılar (konsol):
 *   - Toplam CPU süresi
 *   - Ortalama adım başına CPU süresi
 *   - Gerçek-simülasyon süresi oranı (RTF)
 *   - Constraint violation (son değer & maksimum)
 *   - Enerji hatası (son değer & maksimum)
 *   - Tamamlanan adım sayısı
 *
 * Derleme:
 *   cd build && cmake .. && make pendulum_benchmark
 * Çalıştırma:
 *   ./build/pendulum_benchmark
 */

#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <string>

#include "mb/core/RigidBody.h"
#include "mb/constraints/SphericalJoint.h"
#include "mb/solvers/NewtonRaphson.h"
#include "mb/integrators/RungeKutta.h"
#include "mb/integrators/BDF.h"
#include "mb/system/MultibodySystem.h"
#include "mb/core/ThreadConfig.h"

using namespace mb;

// ── Sistem sabitleri ──────────────────────────────────────────────────────────
static constexpr double L1  = 1.0;
static constexpr double L2  = 1.0;
static constexpr double R   = 0.02;
static constexpr double m1  = 1.0;
static constexpr double m2  = 1.0;
static constexpr double g   = 9.81;

// ── Benchmark parametreleri ───────────────────────────────────────────────────
static constexpr double DT          = 0.005;   // adım [s]
static constexpr double T_END       = 30.0;
static constexpr double HHT_ALPHA   = -0.04;  // HHT sönümleme: [-1/3, 0]
static constexpr double HHT_DT      = 0.005; // enerji/constraint dengesi için varsayılan
static constexpr double HHT_RELTOL  = 1e-4;   // Newton göreli tolerans
static constexpr double HHT_ABSTOL  = 1e-6;   // Newton mutlak tolerans

// ─────────────────────────────────────────────────────────────
//  Simülasyon yapısı — pendulum_qt.cpp ile birebir aynı sistem
// ─────────────────────────────────────────────────────────────
struct Simulation {
    MultibodySystem sys{"DoublePendulum"};

    std::shared_ptr<RigidBody> ground, link1, link2;
    std::shared_ptr<SphericalJoint> joint1, joint2;

    void build(double omega0 = 3.0) {
        sys = MultibodySystem("DoublePendulum");
        sys.setGravity(Vec3(0, -g, 0));

        ground = RigidBody::createGround("Ground");
        sys.addBody(ground);

        // ── Link 1 ──────────────────────────────────────────────────────────
        link1 = RigidBody::createRod(m1, L1, R, "Link1");
        link1->position    = Vec3(L1 * 0.5, 0, 0);
        link1->orientation = Quaternion(0.7071067811865476, 0, 0, -0.7071067811865476);
        link1->angularVelocity = Vec3::zero();
        link1->velocity        = Vec3::zero();
        sys.addBody(link1);

        // ── Link 2 ──────────────────────────────────────────────────────────
        link2 = RigidBody::createRod(m2, L2, R, "Link2");
        link2->position        = Vec3(L1 + L2 * 0.5, 0, 0);
        link2->orientation     = Quaternion(0.7071067811865476, 0, 0, -0.7071067811865476);
        link2->angularVelocity = Vec3::zero();
        link2->velocity        = Vec3::zero();
        sys.addBody(link2);

        // ── Constraints ─────────────────────────────────────────────────────
        joint1 = std::make_shared<SphericalJoint>(
            ground.get(), link1.get(),
            Vec3(0, 0, 0), Vec3(0, -L1 * 0.5, 0), "J1");
        sys.addConstraint(joint1);

        joint2 = std::make_shared<SphericalJoint>(
            link1.get(), link2.get(),
            Vec3(0, L1 * 0.5, 0), Vec3(0, -L2 * 0.5, 0), "J2");
        sys.addConstraint(joint2);

        SolverConfig solverCfg;
        solverCfg.maxIterations = 25;
        solverCfg.tolerance = 1e-10;
        solverCfg.warmStart = true;
        sys.setSolver(std::make_shared<NewtonRaphsonSolver>(solverCfg));

        // integrator seçimi dışarıdan yapılır
        sys.initialize();
    }

    void useHHT(double alpha, double dt) {
        mb::IntegratorConfig cfg;
        cfg.adaptive = false;
        cfg.relTol   = HHT_RELTOL;
        cfg.absTol   = HHT_ABSTOL;
        cfg.maxStep  = dt;
        cfg.minStep  = std::max(1e-8, dt * 1e-3);
        sys.setIntegrator(std::make_shared<HHTAlpha>(alpha, 50, HHT_ABSTOL, cfg));
    }

    void useDOPRI(double relTol = 1e-8, double absTol = 1e-10) {
        mb::IntegratorConfig cfg;
        cfg.adaptive = true;
        cfg.relTol   = relTol;
        cfg.absTol   = absTol;
        sys.setIntegrator(std::make_shared<DormandPrince45>(cfg));
    }

    void step() { sys.step(DT); }
};

// ─────────────────────────────────────────────────────────────
//  Yardımcı: güzel formatlı ayraç
// ─────────────────────────────────────────────────────────────
static void printSeparator(char c = '-', int w = 60) {
    for (int i = 0; i < w; i++) putchar(c);
    putchar('\n');
}

// ─────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {

    bool useDopri = false;
    double dopriRel = 1e-8, dopriAbs = 1e-10;
    double hhtAlpha = HHT_ALPHA;
    double dtStep = HHT_DT;
    double simDuration = T_END;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-c" && i + 1 < argc)
            ThreadConfig::setNumThreads(std::atoi(argv[++i]));
        else if (a == "--dopri") useDopri = true;
        else if (a == "--hht")   useDopri = false;
        else if (a == "--alpha" && i + 1 < argc) hhtAlpha = std::atof(argv[++i]);
        else if (a == "--dt" && i + 1 < argc)    dtStep = std::atof(argv[++i]);
        else if (a == "--tend" && i + 1 < argc)  simDuration = std::atof(argv[++i]);
    }

    if (hhtAlpha < -1.0/3.0) hhtAlpha = -1.0/3.0;
    if (hhtAlpha > 0.0) hhtAlpha = 0.0;
    if (dtStep <= 0.0) dtStep = HHT_DT;
    if (simDuration <= 0.0) simDuration = T_END;

    printSeparator('=');
    printf("  MBC++ — Double Pendulum Benchmark\n");
    printf("  Sistem : L1=%.1f m  L2=%.1f m  m1=%.1f kg  m2=%.1f kg  g=%.2f m/s²\n",
           L1, L2, m1, m2, g);
    printf("  T_end  : %.1f s\n", simDuration);
    if (useDopri) {
        printf("  Integratör : DormandPrince45 (adaptif)\n");
        printf("  relTol=%.1e  absTol=%.1e\n", dopriRel, dopriAbs);
    } else {
        printf("  Integratör : HHT-α (implicit, sabit adım)\n");
        printf("  dt=%.4f s  alpha=%.4f  absTol=%.1e\n", dtStep, hhtAlpha, HHT_ABSTOL);
    }
    printSeparator('=');
    fflush(stdout);

    // ── Kurulum ────────────────────────────────────────────────────────────
    Simulation sim;
    sim.build(3.0);
    if (useDopri) sim.useDOPRI(dopriRel, dopriAbs);
    else          sim.useHHT(hhtAlpha, dtStep);

    auto analyzeInitial = sim.sys.analyze();
    double E0 = analyzeInitial.totalEnergy;

    printf("  Başlangıç enerjisi: %.10f J\n", E0);
    printSeparator();

    // ── Benchmark döngüsü ──────────────────────────────────────────────────
    double maxConstraintViolation = 0.0;
    double maxEnergyError         = 0.0;
    long   totalSteps             = 0;

    // İlerleme çubuğu için
    constexpr int   BAR_WIDTH     = 40;
    int             lastBarFilled = -1;

    using Clock = std::chrono::high_resolution_clock;
    auto wallStart = Clock::now();

    while (sim.sys.getTime() < simDuration) {
        if (useDopri) sim.step();
        else sim.sys.step(dtStep);
        totalSteps++;

        auto a = sim.sys.analyze();

        double cv = a.constraintViolation;
        double ee = std::abs(a.totalEnergy - E0);

        if (cv > maxConstraintViolation) maxConstraintViolation = cv;
        if (ee > maxEnergyError)         maxEnergyError = ee;

        // İlerleme çubuğu (wall-clock'a göre değil, simülasyon zamanına göre)
        double progress = sim.sys.getTime() / simDuration;
        int barFilled   = static_cast<int>(progress * BAR_WIDTH);
        if (barFilled != lastBarFilled) {
            lastBarFilled = barFilled;
            printf("\r  [");
            for (int i = 0; i < BAR_WIDTH; i++)
                putchar(i < barFilled ? '#' : '.');
            printf("] %5.1f%%  t=%.3fs", progress * 100.0, sim.sys.getTime());
            fflush(stdout);
        }
    }

    auto wallEnd = Clock::now();
    double wallSec = std::chrono::duration<double>(wallEnd - wallStart).count();

    // ── Son analiz ─────────────────────────────────────────────────────────
    auto aFinal = sim.sys.analyze();
    double cvFinal = aFinal.constraintViolation;
    double eeFinal = std::abs(aFinal.totalEnergy - E0);

    printf("\n");
    printSeparator('=');
    printf("  BENCHMARK SONUÇLARI\n");
    printSeparator('=');

    printf("\n  ── Zamanlama ───────────────────────────────────────────\n");
    printf("  Toplam wall-clock süresi  : %10.4f s\n",  wallSec);
    printf("  Simülasyon süresi         : %10.4f s\n",  sim.sys.getTime());
    printf("  Gerçek-zaman faktörü (RTF): %10.4f x  (RTF>1 → gerçek zamandan hızlı)\n",
           sim.sys.getTime() / wallSec);
    printf("  Adım başına ortalama süre : %10.4f µs\n",
           wallSec / totalSteps * 1e6);
    printf("  Toplam adım sayısı        : %ld\n", totalSteps);

    printf("\n  ── Constraint Hatası ───────────────────────────────────\n");
    printf("  Son değer                 : %e\n", cvFinal);
    printf("  Maksimum                  : %e\n", maxConstraintViolation);

    printf("\n  ── Enerji Hatası (|E - E0|) ────────────────────────────\n");
    printf("  Başlangıç enerjisi E0     : %.10f J\n",  E0);
    printf("  Son toplam enerji         : %.10f J\n",  aFinal.totalEnergy);
    printf("    Son KE / PE             : %.6f / %.6f J\n",
           aFinal.kineticEnergy, aFinal.potentialEnergy);
    printf("  Son enerji hatası         : %e J\n",  eeFinal);
    printf("  Maksimum enerji hatası    : %e J\n",  maxEnergyError);

    printSeparator('=');
    printf("  Benchmark tamamlandı.\n");
    printSeparator('=');

    return 0;
}
