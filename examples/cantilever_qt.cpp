/**
 * ANCF Cantilever Beam — Flexible multibody dynamics demo with Qt.
 *
 * A steel beam (1 m × 0.05 m × 0.05 m) is clamped at x = 0 and
 * falls under gravity.  Uses the Dormand-Prince 4(5) adaptive
 * integrator for accurate, stable time integration.
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
static constexpr double BEAM_Lz   = 1;    // depth  (m)
static constexpr int    MESH_NX   =5;
static constexpr int    MESH_NY   = 1;
static constexpr int    MESH_NZ   = 1;
static bool             USE_HEX_MESH = true;
static bool             USE_HHT_SOLVER = false;

// Material (Neo-Hookean — stable under large deformation)
// E=2e8 Pa gives ~23 cm static tip deflection (visible but no element inversion)
static constexpr double MAT_E     = 70e9;     // Pa
static constexpr double MAT_NU    = 0.3;
static constexpr double MAT_RHO   = 7800.0;   // kg/m³

// ─────────────────────────────────────────────
//  Simulation state
// ─────────────────────────────────────────────
struct Simulation {
    std::shared_ptr<FlexibleBody> body;
    std::unique_ptr<FlexHHTIntegrator> hht;
    std::unique_ptr<ImplicitFlexIntegrator> implicitInt;

    double dtFrame   = 0.005;     // 1 ms per frame (smaller for stability with full ANCF)
    double time      = 0;

    FlexStepResult lastResult{};

    // Cached rendering data
    std::vector<Vec3>               nodePos;
    std::vector<std::array<int,4>>  tetConn;
    std::vector<std::array<int,8>>  hexConn;
    std::vector<double>             elemStrain; // per-element strain energy

    void build() {
        auto mesh = USE_HEX_MESH
            ? generateBoxHexMesh(BEAM_Lx, BEAM_Ly, BEAM_Lz, MESH_NX, MESH_NY, MESH_NZ)
            : generateBoxTetMesh(BEAM_Lx, BEAM_Ly, BEAM_Lz, MESH_NX, MESH_NY, MESH_NZ);

        ElasticMaterialProps mat{MAT_E, MAT_NU, MAT_RHO, MaterialType::NeoHookean};
        body = FlexibleBody::fromMesh(mesh, mat, "Cantilever", true);

        body->gravity = Vec3(0, -9.81, 0);
        body->dampingAlpha = USE_HHT_SOLVER ? 20.0 : 0.0;

        // Fix the left face (x ≈ 0)
        body->fixNodesOnPlane('x', 0.0, 1e-6);

        if (USE_HHT_SOLVER) {
            hht = std::make_unique<FlexHHTIntegrator>(*body);
            hht->alpha = -0.3;           // strong numerical damping for stability
            hht->newtonTol = 1e-3;
            hht->maxNewtonIter = 5;
            hht->useAnalyticStiffness = true;   // analytic stiffness with penalty
            hht->verbose = false;
            implicitInt.reset();
        } else {
            implicitInt = std::make_unique<ImplicitFlexIntegrator>(*body);
            implicitInt->hhtAlpha = 0.0;
            implicitInt->newtonTol = 1e-6;
            implicitInt->maxNewtonIter = 20;
            implicitInt->fdEps = 1e-6;
            hht.reset();
        }

        time = 0;
        updateRenderData();
    }

    void step() {
        lastResult = USE_HHT_SOLVER ? hht->step(dtFrame) : implicitInt->step(dtFrame);
        time += dtFrame;
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
        // Normalise
        if (maxSE > 1e-20)
            for (auto& v : elemStrain) v /= maxSE;
    }
};

// ─────────────────────────────────────────────
//  Colour map: blue → cyan → green → yellow → red
// ─────────────────────────────────────────────
static QColor heatmap(double t) {
    t = std::clamp(t, 0.0, 1.0);
    // 5-segment piecewise
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
class CantileverWidget : public QWidget {
public:
    CantileverWidget(bool record = false, QWidget* parent = nullptr)
        : QWidget(parent), recording_(record) {
        setWindowTitle("MBC++ — ANCF Cantilever Beam");
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
                 << "cantilever_sim.mp4";
            ffmpeg_->start("ffmpeg", args);
            ffmpeg_->waitForStarted();
            printf("[REC] Recording to cantilever_sim.mp4 (%dx%d @60fps)\n", width(), height());
        }

        timer_ = new QTimer(this);
        connect(timer_, &QTimer::timeout, this, &CantileverWidget::tick);
        timer_->start(16);

        elapsed_.start();
        setFocusPolicy(Qt::StrongFocus);
    }

    ~CantileverWidget() override {
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
            // rotate around Y (azimuth), then around X (elevation)
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
            drawAxis(Vec3(0, 0, 0.15), QColor(70, 130, 255), "Z");
        }

        // ── Wall (clamp) ──
        {
            QPointF top  = toScreen(Vec3(0,  BEAM_Ly * 3, 0));
            QPointF bot  = toScreen(Vec3(0, -BEAM_Ly * 3, 0));
            p.setPen(QPen(QColor(160, 160, 160), 2));
            p.drawLine(top, bot);
            for (int i = -4; i <= 4; i++) {
                double yy = i * BEAM_Ly * 0.75;
                QPointF a = toScreen(Vec3(0, yy, 0));
                QPointF b = toScreen(Vec3(-BEAM_Lx * 0.03, yy - BEAM_Ly * 0.6, 0));
                p.drawLine(a, b);
            }
        }

        // ── Draw filled tetrahedra (projected to XY) ──
        auto& np = sim_.nodePos;
        auto& tc = sim_.tetConn;
        auto& hc = sim_.hexConn;
        auto& se = sim_.elemStrain;
        int nTet = (int)tc.size();

        // Each tet has 4 triangular faces; draw the 4 faces as filled triangles
        for (int e = 0; e < nTet; e++) {
            QColor col = heatmap(se[e]);
            col.setAlpha(180);

            int n0 = tc[e][0], n1 = tc[e][1], n2 = tc[e][2], n3 = tc[e][3];
            QPointF pts[4] = {toScreen(np[n0]), toScreen(np[n1]),
                              toScreen(np[n2]), toScreen(np[n3])};

            // Draw 4 faces of tetrahedron
            int faces[4][3] = {{0,1,2},{0,1,3},{0,2,3},{1,2,3}};
            for (auto& f : faces) {
                QPainterPath path;
                path.moveTo(pts[f[0]]);
                path.lineTo(pts[f[1]]);
                path.lineTo(pts[f[2]]);
                path.closeSubpath();
                p.fillPath(path, col);
            }
        }

        // ── Draw edges ──
        p.setPen(QPen(QColor(200, 210, 230, 100), 0.8));
        for (int e = 0; e < nTet; e++) {
            int ids[4] = {tc[e][0], tc[e][1], tc[e][2], tc[e][3]};
            QPointF pts[4];
            for (int k = 0; k < 4; k++) pts[k] = toScreen(np[ids[k]]);
            for (int a = 0; a < 4; a++)
                for (int b = a+1; b < 4; b++)
                    p.drawLine(pts[a], pts[b]);
        }

        // ── Draw hexahedra edges ──
        if (!hc.empty()) {
            p.setPen(QPen(QColor(230, 210, 120, 180), 1.0));
            int edgePairs[12][2] = {
                {0,1},{1,2},{2,3},{3,0},
                {4,5},{5,6},{6,7},{7,4},
                {0,4},{1,5},{2,6},{3,7}
            };
            for (int e = 0; e < (int)hc.size(); e++) {
                QPointF pts[8];
                for (int k = 0; k < 8; k++) pts[k] = toScreen(np[hc[e][k]]);
                for (auto& edge : edgePairs)
                    p.drawLine(pts[edge[0]], pts[edge[1]]);
            }
        }

        // ── Draw nodes ──
        for (size_t i = 0; i < np.size(); i++) {
            QPointF pt = toScreen(np[i]);
            bool fixed = sim_.body->nodes[i].fixed;
            if (fixed) {
                p.setPen(Qt::NoPen);
                p.setBrush(QColor(255, 80, 80, 200));
                p.drawEllipse(pt, 4, 4);
            } else {
                p.setPen(Qt::NoPen);
                p.setBrush(QColor(200, 220, 255, 160));
                p.drawEllipse(pt, 2.5, 2.5);
            }
        }

        // ── Colour bar ──
        drawColourBar(p, w, h);

        // ── HUD ──
        drawHUD(p, w, h);
    }

    void keyPressEvent(QKeyEvent* e) override {
        switch (e->key()) {
        case Qt::Key_Space:
            paused_ = !paused_;
            break;
        case Qt::Key_R:
            sim_.build();
            elapsed_.restart();
            frameCount_ = 0;
            azimuth_ = elevation_ = panX_ = panY_ = 0;
            zoom_ = 1.0;
            break;
        case Qt::Key_Plus: case Qt::Key_Equal:
            zoom_ *= 1.2;
            break;
        case Qt::Key_Minus:
            zoom_ /= 1.2;
            break;
        case Qt::Key_Escape:
            stopRecording();
            close();
            break;
        }
    }

    void mousePressEvent(QMouseEvent* e) override {
        lastMousePos_ = e->pos();
        e->accept();
    }

    void mouseMoveEvent(QMouseEvent* e) override {
        QPoint d = e->pos() - lastMousePos_;
        lastMousePos_ = e->pos();
        if (e->buttons() & Qt::LeftButton) {
            azimuth_   -= d.x() * 0.005;
            elevation_ -= d.y() * 0.005;
            elevation_  = std::clamp(elevation_, -1.5, 1.5);
        }
        if (e->buttons() & (Qt::RightButton | Qt::MiddleButton)) {
            panX_ += d.x();
            panY_ += d.y();
        }
        update();
        e->accept();
    }

    void wheelEvent(QWheelEvent* e) override {
        double f = (e->angleDelta().y() > 0) ? 1.15 : 1.0 / 1.15;
        zoom_ *= f;
        update();
        e->accept();
    }

private slots:
    void tick() {
        if (!paused_) sim_.step();
        frameCount_++;
        update();

        if (recording_ && ffmpeg_ && ffmpeg_->state() == QProcess::Running) {
            QImage img(size(), QImage::Format_ARGB32);
            img.fill(Qt::black);
            render(&img);
            ffmpeg_->write((const char*)img.constBits(), img.sizeInBytes());
        }
    }

private:
    Simulation sim_;
    QTimer* timer_ = nullptr;
    QElapsedTimer elapsed_;
    bool paused_    = false;
    int  frameCount_ = 0;
    bool recording_  = false;
    QProcess* ffmpeg_ = nullptr;
    double zoom_      = 1.0;
    double azimuth_   = 0.0;   // orbit around Y (rad)
    double elevation_ = 0.0;   // orbit around X (rad)
    double panX_      = 0.0;   // pan offset (px)
    double panY_      = 0.0;
    QPoint lastMousePos_;

    void stopRecording() {
        if (ffmpeg_ && ffmpeg_->state() == QProcess::Running) {
            ffmpeg_->closeWriteChannel();
            ffmpeg_->waitForFinished(5000);
            printf("[REC] Saved cantilever_sim.mp4\n");
            ffmpeg_ = nullptr;
        }
    }

    void drawColourBar(QPainter& p, int w, int h) {
        int bx = w - 45, by = h/2 - 100, bw = 18, bh = 200;
        for (int i = 0; i < bh; i++) {
            double t = 1.0 - double(i) / bh;
            p.setPen(heatmap(t));
            p.drawLine(bx, by + i, bx + bw, by + i);
        }
        p.setPen(QColor(140, 140, 160));
        p.drawRect(bx - 1, by - 1, bw + 2, bh + 2);

        QFont f("Sans", 8);
        p.setFont(f);
        p.drawText(bx - 10, by - 8,  "high");
        p.drawText(bx - 8,  by + bh + 14, "low");
        // Vertical label
        p.save();
        p.translate(bx + bw + 16, by + bh/2);
        p.rotate(-90);
        p.drawText(0, 0, "Strain Energy");
        p.restore();
    }

    void drawHUD(QPainter& p, int w, int /*h*/) {
        double fps = frameCount_ / (elapsed_.elapsed() * 0.001 + 1e-9);

        p.setPen(QColor(200, 200, 200));
        QFont f("Monospace", 11);
        f.setStyleHint(QFont::Monospace);
        p.setFont(f);

        int x = 14, y = 24, dy = 20;
        auto line = [&](const QString& s) {
            p.drawText(x, y, s);
            y += dy;
        };

        line(QString("t = %1 s").arg(sim_.time, 0, 'f', 4));
        line(QString("FPS  %1").arg(fps, 0, 'f', 1));
        if (USE_HHT_SOLVER) {
            line(QString("Integrator: HHT-α (α=%1)").arg(sim_.hht->alpha, 0, 'f', 2));
            line(QString("Newton iters: %1  |R|=%2").arg(sim_.hht->lastNewtonIters()).arg(sim_.hht->lastResidualNorm(), 0, 'e', 2));
            line(QString("Total steps: %1").arg(sim_.hht->totalSteps()));
        } else {
            line(QString("Integrator: Implicit Newmark (α=%1)").arg(sim_.implicitInt->hhtAlpha, 0, 'f', 2));
            line(QString("Newton iters: %1  |R|=%2").arg(sim_.implicitInt->lastNewtonIters()).arg(sim_.implicitInt->lastResidualNorm(), 0, 'e', 2));
            line(QString("Total steps: %1").arg(sim_.implicitInt->totalSteps()));
        }
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
};

// ─────────────────────────────────────────────
int main(int argc, char* argv[]) {
    // Parse -c N for thread count, -r for recording, --hex for hexa mesh
    bool record = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-c" && i+1 < argc) {
            ThreadConfig::setNumThreads(std::atoi(argv[++i]));
        } else if (std::string(argv[i]) == "-r") {
            record = true;
        } else if (std::string(argv[i]) == "--hex") {
            USE_HEX_MESH = true;
        } else if (std::string(argv[i]) == "--hht") {
            USE_HHT_SOLVER = true;
        }
    }

    QApplication app(argc, argv);
    CantileverWidget win(record);
    win.show();
    return app.exec();
}
