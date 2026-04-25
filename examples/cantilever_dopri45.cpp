/**
 * ANCF Cantilever Beam — DOPRI45 Adaptive Integrator Demo
 *
 * A steel beam (1 m × 0.05 m × 0.05 m) is clamped at x = 0 and
 * falls under gravity. Uses the Dormand-Prince 4(5) adaptive integrator
 * for accurate time integration with strict energy conservation.
 *
 * Colour encodes strain energy density at each element.
 *
 * Controls:
 *   Space  — Pause / Resume
 *   R      — Reset
 *   +/-    — Zoom in / out
 *   ESC    — Quit
 */
#include <QApplication>
#include <QWidget>
#include <QPainter>
#include <QPainterPath>
#include <QTimer>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QElapsedTimer>
#include <QFont>
#include <QProcess>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <memory>
#include <iostream>
#include <cstring>

#include "mb/fem/ANCFTypes.h"
#include "mb/fem/FlexibleBody.h"
#include "mb/fem/FlexibleIntegrators.h"
#include "mb/core/ThreadConfig.h"
#include "mb/fem/MeshGenerators.h"

#include <cstdlib>

using namespace mb;

// ─────────────────────────────────────────────
//  Beam parameters
// ─────────────────────────────────────────────
static constexpr double BEAM_Lx   = 1;     // length (m)
static constexpr double BEAM_Ly   = 0.05;    // height (m)
static constexpr double BEAM_Lz   = 0.05;    // depth  (m)
static constexpr int    MESH_NX   = 10;
static constexpr int    MESH_NY   = 1;
static constexpr int    MESH_NZ   = 1;
static bool             USE_HEX_MESH = true;

// Material (Neo-Hookean)
static constexpr double MAT_E     = 70e9;     // Pa
static constexpr double MAT_NU    = 0.3;
static constexpr double MAT_RHO   = 7800.0;   // kg/m³

// ─────────────────────────────────────────────
//  Simulation state
// ─────────────────────────────────────────────
struct Simulation {
    std::shared_ptr<FlexibleBody> body;
    std::unique_ptr<FlexDOPRI45> dopri;

    double dtFrame   = 0.01;      // target 10 ms per frame (adaptive may take smaller steps)
    double time      = 0;
    double timeE0    = 0;         // energy reference time

    FlexStepResult lastResult{};
    double E0        = 0;         // initial total energy
    int stepsTaken   = 0;
    int stepsRejected = 0;

    // Cached rendering data
    std::vector<Vec3>               nodePos;
    std::vector<std::array<int,4>>  tetConn;
    std::vector<std::array<int,8>>  hexConn;
    std::vector<double>             elemStrain;

    void build() {
        auto mesh = USE_HEX_MESH
            ? generateBoxHexMesh(BEAM_Lx, BEAM_Ly, BEAM_Lz, MESH_NX, MESH_NY, MESH_NZ)
            : generateBoxTetMesh(BEAM_Lx, BEAM_Ly, BEAM_Lz, MESH_NX, MESH_NY, MESH_NZ);

        ElasticMaterialProps mat{MAT_E, MAT_NU, MAT_RHO, MaterialType::NeoHookean};
        body = FlexibleBody::fromMesh(mesh, mat, "Cantilever", true);

        body->gravity = Vec3(0, -9.81, 0);
        body->dampingAlpha = 0.0;   // no physical damping — pure energy conservation

        // Fix the left face (x ≈ 0)
        body->fixNodesOnPlane('x', 0.0, 1e-6);

        dopri = std::make_unique<FlexDOPRI45>(*body);
        dopri->absTol  = 1e-6;
        dopri->relTol  = 1e-4;
        dopri->maxStep = 0.01;
        dopri->minStep = 1e-8;
        dopri->dtCurrent = 1e-4;

        time = 0;
        timeE0 = 0;
        stepsTaken = 0;
        stepsRejected = 0;
        E0 = 0;
        updateRenderData();

        // Measure initial energy
        double SE = body->computeStrainEnergy();
        double KE = body->computeKineticEnergy();
        double PE = body->computePotentialEnergy(body->gravity);
        E0 = SE + KE + PE - SE;  // PE - SE (gravitational PE only)
    }

    void step() {
        lastResult = dopri->step(dtFrame);
        time += dtFrame;
        stepsTaken = dopri->totalSteps;
        stepsRejected = dopri->totalRejects;
        updateRenderData();
    }

    void updateRenderData() {
        nodePos = body->getNodePositions();
        tetConn = body->getTetConnectivity();
        hexConn = body->getHexConnectivity();

        // Per-element strain energy (for colouring)
        int nElem = (int)body->elements.size() + (int)body->hexElements.size();
        elemStrain.resize(nElem);
        double maxSE = 0;
        int out = 0;
        for (int e = 0; e < (int)body->elements.size(); e++) {
            auto& elem = body->elements[e];
            double F[9];
            elem.computeDeformationGradient(0.25, 0.25, 0.25, body->nodes, F);
            double se = body->material->strainEnergyDensity(F);
            elemStrain[out] = se * elem.V0;
            maxSE = std::max(maxSE, elemStrain[out]);
            out++;
        }
        for (int e = 0; e < (int)body->hexElements.size(); e++) {
            auto& elem = body->hexElements[e];
            double F[9];
            elem.computeDeformationGradient(0.0, 0.0, 0.0, body->nodes, F);
            double se = body->material->strainEnergyDensity(F);
            elemStrain[out] = se * elem.V0;
            maxSE = std::max(maxSE, elemStrain[out]);
            out++;
        }
        if (maxSE > 1e-20)
            for (auto& v : elemStrain) v /= maxSE;
    }
};

// ─────────────────────────────────────────────
//  Colour map: blue → cyan → green → yellow → red
// ─────────────────────────────────────────────
static QColor heatmap(double t) {
    t = std::clamp(t, 0.0, 1.0);
    double r, g, b;
    if (t < 0.25) {
        double s = t / 0.25;
        r = 0;   g = s;     b = 1;
    } else if (t < 0.5) {
        double s = (t - 0.25) / 0.25;
        r = 0;   g = 1;     b = 1 - s;
    } else if (t < 0.75) {
        double s = (t - 0.5) / 0.25;
        r = s;   g = 1;     b = 0;
    } else {
        double s = (t - 0.75) / 0.25;
        r = 1;   g = 1 - s; b = 0;
    }
    return QColor(int(r*255), int(g*255), int(b*255));
}

// ─────────────────────────────────────────────
//  Qt Widget
// ─────────────────────────────────────────────
class CantileverDOPRI45Widget : public QWidget {
public:
    CantileverDOPRI45Widget(bool record = false, QWidget* parent = nullptr)
        : QWidget(parent), recording_(record) {
        setWindowTitle("MBC++ — ANCF Cantilever DOPRI45");
        resize(1200, 700);
        setMinimumSize(900, 500);

        sim_.build();

        if (recording_) {
            ffmpeg_ = new QProcess(this);
            QStringList args;
            args << "-y" << "-f" << "rawvideo" << "-pixel_format" << "bgra"
                 << "-video_size" << QString("%1x%2").arg(width()).arg(height())
                 << "-framerate" << "60"
                 << "-i" << "pipe:0"
                 << "-c:v" << "libx264" << "-preset" << "fast"
                 << "-crf" << "18" << "-pix_fmt" << "yuv420p"
                 << "cantilever_dopri45.mp4";
            ffmpeg_->start("ffmpeg", args);
            ffmpeg_->waitForStarted();
            printf("[REC] Recording to cantilever_dopri45.mp4 (%dx%d @60fps)\n", width(), height());
        }

        timer_ = new QTimer(this);
        connect(timer_, &QTimer::timeout, this, &CantileverDOPRI45Widget::tick);
        timer_->start(16);

        elapsed_.start();
        setFocusPolicy(Qt::StrongFocus);
    }

    ~CantileverDOPRI45Widget() override {
        stopRecording();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);

        int w = width(), h = height();
        p.fillRect(rect(), QColor(15, 17, 22));

        // ── Camera (3D orbit) ──
        double ppm = zoom_ * std::min(w, h) * 0.6 / BEAM_Lx;
        double cx  = w * 0.35 + panX_;
        double cy  = h * 0.45 + panY_;

        double ca = std::cos(azimuth_), sa = std::sin(azimuth_);
        double cElev = std::cos(elevation_), sElev = std::sin(elevation_);

        auto toScreen = [&](const Vec3& v) -> QPointF {
            double x1 =  v.x * ca + v.z * sa;
            double y1 =  v.y;
            double z1 = -v.x * sa + v.z * ca;
            double x2 =  x1;
            double y2 =  y1 * cElev - z1 * sElev;
            return {cx + x2 * ppm, cy - y2 * ppm};
        };

        // ── Grid (XY plane at z=0) ──
        {
            p.setPen(QPen(QColor(30, 33, 42), 1));
            double step = 0.1;
            if (ppm * step < 15) step = 0.2;
            if (ppm * step < 15) step = 0.5;
            double xlo = -0.5, xhi = 1.5, ylo = -1.5, yhi = 0.5;
            for (double m = xlo; m <= xhi + 1e-9; m += step)
                p.drawLine(toScreen(Vec3(m, ylo, 0)),
                           toScreen(Vec3(m, yhi, 0)));
            for (double m = ylo; m <= yhi + 1e-9; m += step)
                p.drawLine(toScreen(Vec3(xlo, m, 0)),
                           toScreen(Vec3(xhi, m, 0)));
        }

        // ── Coordinate axes ──
        {
            auto drawAxis = [&](Vec3 tip, QColor c, const QString& lbl) {
                p.setPen(QPen(c, 2));
                p.drawLine(toScreen(Vec3(0,0,0)), toScreen(tip));
                p.setFont(QFont("Sans", 9, QFont::Bold));
                p.drawText(toScreen(tip) + QPointF(4, -4), lbl);
            };
            drawAxis(Vec3(0.15, 0, 0), QColor(220, 70, 70),  "X");
            drawAxis(Vec3(0, 0.15, 0), QColor(70, 200, 70),  "Y");
            drawAxis(Vec3(0, 0, 0.15), QColor(70, 120, 200), "Z");
        }

        // ── Render tetrahedral elements ──
        if (!USE_HEX_MESH) {
            for (const auto& conn : sim_.tetConn) {
                std::array<QPointF, 4> pts{
                    toScreen(sim_.nodePos[conn[0]]),
                    toScreen(sim_.nodePos[conn[1]]),
                    toScreen(sim_.nodePos[conn[2]]),
                    toScreen(sim_.nodePos[conn[3]])
                };
                for (int i = 0; i < 4; i++) {
                    for (int j = i+1; j < 4; j++) {
                        int elemIdx = (int)(&conn - sim_.tetConn.data());
                        QColor c = heatmap(sim_.elemStrain[elemIdx]);
                        p.setPen(QPen(c, 1.5));
                        p.drawLine(pts[i], pts[j]);
                    }
                }
            }
        }

        // ── Render hexahedral elements ──
        if (USE_HEX_MESH) {
            for (const auto& conn : sim_.hexConn) {
                std::array<QPointF, 8> pts{
                    toScreen(sim_.nodePos[conn[0]]),
                    toScreen(sim_.nodePos[conn[1]]),
                    toScreen(sim_.nodePos[conn[2]]),
                    toScreen(sim_.nodePos[conn[3]]),
                    toScreen(sim_.nodePos[conn[4]]),
                    toScreen(sim_.nodePos[conn[5]]),
                    toScreen(sim_.nodePos[conn[6]]),
                    toScreen(sim_.nodePos[conn[7]])
                };
                // Hex wireframe: bottom face (0,1,2,3), top (4,5,6,7), verticals
                int elemIdx = (int)(&conn - sim_.hexConn.data());
                QColor c = heatmap(sim_.elemStrain[sim_.tetConn.size() + elemIdx]);
                p.setPen(QPen(c, 1.5));

                // Bottom quad
                p.drawLine(pts[0], pts[1]); p.drawLine(pts[1], pts[2]);
                p.drawLine(pts[2], pts[3]); p.drawLine(pts[3], pts[0]);
                // Top quad
                p.drawLine(pts[4], pts[5]); p.drawLine(pts[5], pts[6]);
                p.drawLine(pts[6], pts[7]); p.drawLine(pts[7], pts[4]);
                // Verticals
                for (int i = 0; i < 4; i++) p.drawLine(pts[i], pts[i+4]);
            }
        }

        // ── HUD ──
        p.setPen(QColor(200, 200, 200));
        QFont f("Monospace", 11);
        f.setStyleHint(QFont::Monospace);
        p.setFont(f);

        int x = 14, y = 24, dy = 20;
        auto line = [&](const QString& s) {
            p.drawText(x, y, s);
            y += dy;
        };

        double fps = 0;
        if (frameTime_ > 1e-9) fps = 1.0 / frameTime_;

        line(QString("t = %1 s").arg(sim_.time, 0, 'f', 4));
        line(QString("FPS  %1").arg(fps, 0, 'f', 1));
        line(QString("Integrator: DOPRI45 (adaptive)"));
        line(QString("Steps: %1  Rejects: %2").arg(sim_.stepsTaken).arg(sim_.stepsRejected));
        line(QString("h_curr: %1 s").arg(sim_.dopri->dtCurrent, 0, 'e', 2));
        line(QString(""));
        line(QString("DOFs: %1  (free %2)")
             .arg(sim_.body->numDof).arg(sim_.body->numFreeDof()));
        line(QString("Nodes: %1  Tets: %2  Hex: %3")
             .arg(sim_.body->nodes.size())
             .arg(sim_.body->elements.size())
             .arg(sim_.body->hexElements.size()));
        line(QString("Mesh mode: %1").arg(USE_HEX_MESH ? "hex" : "tet"));
        line(QString(""));
        line(QString("Max disp: %1 mm").arg(sim_.lastResult.maxDisplacement * 1e3, 0, 'f', 2));
        line(QString("Strain E: %1 J").arg(sim_.lastResult.strainEnergy, 0, 'e', 3));
        line(QString("Kinetic E: %1 J").arg(sim_.lastResult.kineticEnergy, 0, 'e', 3));
        line(QString("Grav.  PE: %1 J").arg(sim_.lastResult.gravitationalPE, 0, 'e', 3));
        double totalE = sim_.lastResult.strainEnergy + sim_.lastResult.kineticEnergy
                      + sim_.lastResult.gravitationalPE;
        line(QString("Total  E: %1 J").arg(totalE, 0, 'e', 3));

        if (sim_.E0 != 0) {
            double dE = totalE - sim_.E0;
            double relErr = dE / std::abs(sim_.E0);
            line(QString("ΔE/E0: %1").arg(relErr, 0, 'e', 3));
        }

        // ── Beam dimensions ──
        y += 10;
        p.setPen(QColor(140, 150, 170));
        QFont f2("Sans", 9);
        p.setFont(f2);
        line(QString("Beam: %1×%2×%3 m")
             .arg(BEAM_Lx).arg(BEAM_Ly).arg(BEAM_Lz));
        line(QString("E = %1 Pa,  ν = %2,  ρ = %3")
             .arg(MAT_E, 0, 'e', 1).arg(MAT_NU).arg(MAT_RHO));

        // Controls (right side)
        p.setPen(QColor(120, 120, 140));
        int bx = w - 270, by = 24;
        auto rline = [&](const QString& s) { p.drawText(bx, by, s); by += 18; };
        rline("Sol sürükle    Döndür");
        rline("Sağ sürükle    Kaydır");
        rline("Tekerlek       Yakınlaştır");
        rline("SPACE          Duraklat");
        rline("R              Sıfırla");
        rline("ESC            Çıkış");

        if (paused_) {
            p.setPen(QColor(255, 80, 80));
            QFont fb("Sans", 16, QFont::Bold);
            p.setFont(fb);
            p.drawText(w / 2 - 50, 40, "⏸  DURAKLATILDI");
        }
    }

    void keyPressEvent(QKeyEvent* e) override {
        if (e->key() == Qt::Key_Escape) QApplication::quit();
        else if (e->key() == Qt::Key_Space) paused_ = !paused_;
        else if (e->key() == Qt::Key_R) { sim_.build(); update(); }
        else if (e->key() == Qt::Key_Plus || e->key() == Qt::Key_Equal) {
            zoom_ *= 1.2; update();
        } else if (e->key() == Qt::Key_Minus) {
            zoom_ /= 1.2; update();
        }
    }

    void mousePressEvent(QMouseEvent* e) override {
        lastMouse_ = e->pos();
    }

    void mouseMoveEvent(QMouseEvent* e) override {
        if (!(e->buttons() & (Qt::LeftButton | Qt::RightButton))) return;
        QPointF delta = e->pos() - lastMouse_;
        if (e->buttons() & Qt::LeftButton) {
            azimuth_ -= delta.x() * 0.01;
            elevation_ += delta.y() * 0.01;
        } else {
            panX_ += delta.x();
            panY_ += delta.y();
        }
        lastMouse_ = e->pos();
        update();
    }

    void wheelEvent(QWheelEvent* e) override {
        double factor = (e->angleDelta().y() > 0) ? 1.2 : 0.833;
        zoom_ *= factor;
        update();
    }

private:
    Simulation sim_;
    QTimer* timer_ = nullptr;
    QProcess* ffmpeg_ = nullptr;
    QElapsedTimer elapsed_;
    double frameTime_ = 0;

    // View
    double zoom_ = 1.0, azimuth_ = 0.3, elevation_ = 0.15;
    double panX_ = 0, panY_ = 0;
    QPointF lastMouse_;

    // State
    bool paused_ = false, recording_ = false;

    void tick() {
        if (!paused_) sim_.step();

        if (recording_) {
            QImage img(size(), QImage::Format_ARGB32);
            QPainter p(&img);
            paintEvent(nullptr);
            QByteArray raw;
            raw.resize(width() * height() * 4);
            memcpy(raw.data(), img.bits(), raw.size());
            ffmpeg_->write(raw);
        }

        update();
        frameTime_ = elapsed_.nsecsElapsed() / 1e9;
        elapsed_.start();
    }

    void stopRecording() {
        if (ffmpeg_ && ffmpeg_->state() == QProcess::Running) {
            ffmpeg_->closeWriteChannel();
            ffmpeg_->waitForFinished(2000);
        }
    }
};

// ─────────────────────────────────────────────
int main(int argc, char* argv[]) {
    bool record = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-c" && i+1 < argc) {
            ThreadConfig::setNumThreads(std::atoi(argv[++i]));
        } else if (std::string(argv[i]) == "-r") {
            record = true;
        } else if (std::string(argv[i]) == "--hex") {
            USE_HEX_MESH = true;
        }
    }

    QApplication app(argc, argv);
    CantileverDOPRI45Widget win(record);
    win.show();
    return app.exec();
}
