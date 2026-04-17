/**
 * ANCF Spinning Beam — Flexible multibody dynamics demo with Qt.
 *
 * A beam is clamped at one end (x = 0) and spins around the Z axis at a
 * prescribed angular velocity ω.  Centrifugal stiffening is clearly visible:
 * at low ω the beam droops under gravity, at high ω it straightens.
 *
 * The centrifugal force is applied each step via the externalForces callback.
 *
 * Controls:
 *   Space  — Pause / Resume
 *   R      — Reset
 *   W / S  — Increase / Decrease angular velocity ω
 *   G      — Toggle gravity (on / off)
 *   +/-    — Zoom in / out
 *   ESC    — Quit
 *
 * Mouse:
 *   Left drag   — Orbit (azimuth + elevation)
 *   Right drag  — Pan
 *   Wheel       — Zoom
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
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <memory>
#include <iostream>

#include "mb/fem/ANCFTypes.h"
#include "mb/fem/FlexibleBody.h"
#include "mb/fem/FlexibleIntegrators.h"

using namespace mb;

// ─────────────────────────────────────────────
//  Beam parameters
// ─────────────────────────────────────────────
static constexpr double BEAM_Lx   = 1.0;    // length along spin radius (m)
static constexpr double BEAM_Ly   = 0.05;   // height (m)
static constexpr double BEAM_Lz   = 0.05;   // depth  (m)
static constexpr int    MESH_NX   = 5;
static constexpr int    MESH_NY   = 1;
static constexpr int    MESH_NZ   = 1;

// Material (Neo-Hookean — stable under large deformation)
static constexpr double MAT_E     = 70e9;    // Pa  — soft enough to show flex
static constexpr double MAT_NU    = 0.3;
static constexpr double MAT_RHO   = 7800.0; // kg/m³

// Initial angular velocity around Z axis (rad/s)
static constexpr double OMEGA_INIT = 5.0;

// ─────────────────────────────────────────────
//  Simulation state
// ─────────────────────────────────────────────
struct Simulation {
    std::shared_ptr<FlexibleBody> body;
    std::unique_ptr<FlexDOPRI45> dopri;

    double dtFrame       = 0.01;   // wall-time per frame (s)
    double time         = 0.0;
    double omega        = OMEGA_INIT; // current spin rate (rad/s)
    double theta        = 0.0;        // current lab-frame rotation angle (rad)
    bool   gravityEnabled = true;     // gravity on/off

    FlexStepResult lastResult{};

    // Cached rendering data
    std::vector<Vec3>               nodePos;
    std::vector<std::array<int,4>>  tetConn;
    std::vector<double>             elemStrain; // per-element normalised strain

    void build() {
        auto mesh = generateBoxTetMesh(BEAM_Lx, BEAM_Ly, BEAM_Lz,
                                       MESH_NX, MESH_NY, MESH_NZ);

        ElasticMaterialProps mat{MAT_E, MAT_NU, MAT_RHO, MaterialType::NeoHookean};
        body = FlexibleBody::fromMesh(mesh, mat, "SpinningBeam", true);

        // All fictitious forces handled in externalForces — keep body gravity zero.
        body->gravity      = Vec3(0, 0, 0);
        body->dampingAlpha = 0.0;

        // Clamp at x = 0  (root attached to hub)
        body->fixNodesOnPlane('x', 0.0, 1e-6);

        // Simulation runs in the rotating (body-fixed) frame.
        // Fictitious forces:
        //   Centrifugal : f = m·ω²·r  (outward radial)
        //   Coriolis    : f = -2m(ω×v) = (2mω·vy, -2mω·vx, 0)
        //   Gravity     : world g=(0,-g,0) transformed to rotating frame
        //                 g_body = R(-θ)·g_world
        body->externalForces = [this](FlexibleBody& b) -> std::vector<double> {
            std::vector<double> F(b.numDof, 0.0);
            auto mdiag = b.getMassDiagonal();
            int nn = (int)b.nodes.size();
            double w2 = omega * omega;
            // Gravity vector in rotating frame
            double gx = 0.0, gy = 0.0;
            if (gravityEnabled) {
                // R(-θ)·(0, -9.81, 0)
                gx = -9.81 * (-std::sin(theta));   // = 9.81*sin(θ) ... wait:
                // R(-θ) = [[cosθ, sinθ],[−sinθ, cosθ]]
                // gx = cosθ*0 + sinθ*(−9.81) = −9.81*sinθ
                // gy = −sinθ*0 + cosθ*(−9.81) = −9.81*cosθ
                gx = -9.81 * std::sin(theta);
                gy = -9.81 * std::cos(theta);
            }
            for (int i = 0; i < nn; i++) {
                auto& nd = b.nodes[i];
                double mi = mdiag[i * 12 + 0];
                double rx = nd.q[0],  ry = nd.q[1];
                double vx = nd.qd[0], vy = nd.qd[1];
                // Centrifugal
                F[i * 12 + 0] += mi * w2 * rx;
                F[i * 12 + 1] += mi * w2 * ry;
                // Coriolis: -2m(ω×v),  ω=(0,0,ω)  →  f=(2mω·vy, -2mω·vx, 0)
                F[i * 12 + 0] += 2.0 * mi * omega * vy;
                F[i * 12 + 1] -= 2.0 * mi * omega * vx;
                // Gravity in rotating frame
                F[i * 12 + 0] += mi * gx;
                F[i * 12 + 1] += mi * gy;
            }
            return F;
        };

        dopri = std::make_unique<FlexDOPRI45>(*body);
        dopri->absTol    = 1e-8;
        dopri->relTol    = 1e-6;
        dopri->maxStep   = 0.005;
        dopri->dtCurrent = 1e-4;

        time  = 0;
        theta = 0;
        updateRenderData();
    }

    void applyGravity() {
        // Gravity is handled inside externalForces in the rotating frame;
        // toggling gravityEnabled is sufficient — no body->gravity change needed.
    }

    void step() {
        lastResult  = dopri->step(dtFrame);
        time  += dtFrame;
        theta += omega * dtFrame;   // advance lab-frame rotation angle
        updateRenderData();
    }

    void updateRenderData() {
        // Body-frame positions → rotate by θ to get lab-frame for rendering
        auto rawPos = body->getNodePositions();
        double cth = std::cos(theta), sth = std::sin(theta);
        nodePos.resize(rawPos.size());
        for (size_t i = 0; i < rawPos.size(); i++) {
            double bx = rawPos[i].x, by = rawPos[i].y;
            nodePos[i].x = bx * cth - by * sth;
            nodePos[i].y = bx * sth + by * cth;
            nodePos[i].z = rawPos[i].z;
        }
        tetConn = body->getTetConnectivity();

        int nElem = (int)body->elements.size();
        elemStrain.resize(nElem);
        double maxSE = 0;
        for (int e = 0; e < nElem; e++) {
            auto& elem = body->elements[e];
            double F[9];
            elem.computeDeformationGradient(0.25, 0.25, 0.25, body->nodes, F);
            double se = body->material->strainEnergyDensity(F);
            elemStrain[e] = se * elem.V0;
            maxSE = std::max(maxSE, elemStrain[e]);
        }
        if (maxSE > 1e-20)
            for (auto& v : elemStrain) v /= maxSE;
    }
};

// ─────────────────────────────────────────────
//  Colour map
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
class SpinningBeamWidget : public QWidget {
public:
    SpinningBeamWidget(QWidget* parent = nullptr) : QWidget(parent) {
        setWindowTitle("MBC++ — ANCF Spinning Beam");
        resize(1200, 700);
        setMinimumSize(900, 500);

        sim_.build();

        timer_ = new QTimer(this);
        connect(timer_, &QTimer::timeout, this, &SpinningBeamWidget::tick);
        timer_->start(16);

        elapsed_.start();
        setFocusPolicy(Qt::StrongFocus);

        // Default: top-down view to see XY spinning plane
        elevation_ = M_PI / 2.0 * 0.85;
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);

        int w = width(), h = height();
        p.fillRect(rect(), QColor(15, 17, 22));

        // ── Camera (3D orbit) ──
        double ppm = zoom_ * std::min(w, h) * 0.55 / BEAM_Lx;
        double cx  = w * 0.5 + panX_;
        double cy  = h * 0.5 + panY_;

        double ca    = std::cos(azimuth_),   sa    = std::sin(azimuth_);
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
            double ext = 1.6;
            for (double m = -ext; m <= ext + 1e-9; m += step)
                p.drawLine(toScreen(Vec3(m, -ext, 0)),
                           toScreen(Vec3(m,  ext, 0)));
            for (double m = -ext; m <= ext + 1e-9; m += step)
                p.drawLine(toScreen(Vec3(-ext, m, 0)),
                           toScreen(Vec3( ext, m, 0)));
        }

        // ── Rotation circle (radius = beam length) ──
        {
            p.setPen(QPen(QColor(60, 70, 100, 120), 1, Qt::DashLine));
            const int nseg = 80;
            for (int i = 0; i < nseg; i++) {
                double a0 = 2 * M_PI * i       / nseg;
                double a1 = 2 * M_PI * (i + 1) / nseg;
                p.drawLine(toScreen(Vec3(BEAM_Lx * std::cos(a0),
                                        BEAM_Lx * std::sin(a0), 0)),
                           toScreen(Vec3(BEAM_Lx * std::cos(a1),
                                        BEAM_Lx * std::sin(a1), 0)));
            }
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

        // ── Hub (clamp) at origin ──
        {
            QPointF hub = toScreen(Vec3(0, 0, 0));
            p.setBrush(QColor(180, 100, 60, 200));
            p.setPen(QPen(QColor(220, 140, 80), 1.5));
            p.drawEllipse(hub, 7, 7);
        }

        // ── Draw filled tetrahedra ──
        auto& np = sim_.nodePos;
        auto& tc = sim_.tetConn;
        auto& se = sim_.elemStrain;
        int nElem = (int)tc.size();

        for (int e = 0; e < nElem; e++) {
            QColor col = heatmap(se[e]);
            col.setAlpha(180);
            int n0 = tc[e][0], n1 = tc[e][1], n2 = tc[e][2], n3 = tc[e][3];
            QPointF pts[4] = {toScreen(np[n0]), toScreen(np[n1]),
                              toScreen(np[n2]), toScreen(np[n3])};
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
        for (int e = 0; e < nElem; e++) {
            int ids[4] = {tc[e][0], tc[e][1], tc[e][2], tc[e][3]};
            QPointF pts[4];
            for (int k = 0; k < 4; k++) pts[k] = toScreen(np[ids[k]]);
            for (int a = 0; a < 4; a++)
                for (int b = a+1; b < 4; b++)
                    p.drawLine(pts[a], pts[b]);
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
            azimuth_ = 0;
            elevation_ = M_PI / 2.0 * 0.85;
            panX_ = panY_ = 0;
            zoom_ = 1.0;
            break;
        case Qt::Key_W:
            sim_.omega = std::min(sim_.omega + 1.0, 100.0);
            break;
        case Qt::Key_S:
            sim_.omega = std::max(sim_.omega - 1.0, 0.0);
            break;
        case Qt::Key_G:
            sim_.gravityEnabled = !sim_.gravityEnabled;
            sim_.applyGravity();
            break;
        case Qt::Key_Plus: case Qt::Key_Equal:
            zoom_ *= 1.2;
            break;
        case Qt::Key_Minus:
            zoom_ /= 1.2;
            break;
        case Qt::Key_Escape:
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
    }

private:
    Simulation sim_;
    QTimer* timer_     = nullptr;
    QElapsedTimer elapsed_;
    bool   paused_     = false;
    int    frameCount_ = 0;
    double zoom_       = 1.0;
    double azimuth_    = 0.0;
    double elevation_  = 0.0;
    double panX_       = 0.0;
    double panY_       = 0.0;
    QPoint lastMousePos_;

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
        auto line = [&](const QString& s) { p.drawText(x, y, s); y += dy; };

        line(QString("t = %1 s").arg(sim_.time, 0, 'f', 4));
        line(QString("FPS  %1").arg(fps, 0, 'f', 1));
        line(QString("Integrator: DOPRI45 (adaptive)"));
        line(QString("dt_adapt = %1 ms").arg(sim_.dopri->dtCurrent * 1e3, 0, 'f', 4));
        line(QString("steps %1  rejects %2")
             .arg(sim_.dopri->totalSteps).arg(sim_.dopri->totalRejects));
        line(QString(""));
        line(QString("DOFs: %1  (free %2)")
             .arg(sim_.body->numDof).arg(sim_.body->numFreeDof()));
        line(QString("Nodes: %1  Tets: %2")
             .arg(sim_.body->nodes.size()).arg(sim_.body->elements.size()));
        line(QString(""));

        // Angular velocity — highlight
        {
            QFont fb("Monospace", 12, QFont::Bold);
            p.setFont(fb);
            double rpm = sim_.omega * 60.0 / (2.0 * M_PI);
            p.setPen(QColor(100, 220, 255));
            p.drawText(x, y,
                QString("ω = %1 rad/s  (%2 RPM)")
                    .arg(sim_.omega, 0, 'f', 1)
                    .arg(rpm, 0, 'f', 0));
            y += dy + 2;
            p.setFont(f);
            p.setPen(QColor(200, 200, 200));
        }

        line(QString("Gravity: %1").arg(sim_.gravityEnabled ? "ON  (g = 9.81 m/s²)" : "OFF"));
        line(QString(""));
        line(QString("Max disp: %1 mm").arg(sim_.lastResult.maxDisplacement * 1e3, 0, 'f', 2));
        line(QString("Strain E: %1 J").arg(sim_.lastResult.strainEnergy, 0, 'e', 3));
        line(QString("Kinetic E: %1 J").arg(sim_.lastResult.kineticEnergy, 0, 'e', 3));
        line(QString("Grav.  PE: %1 J").arg(sim_.lastResult.gravitationalPE, 0, 'e', 3));
        double totalE = sim_.lastResult.strainEnergy
                      + sim_.lastResult.kineticEnergy
                      + sim_.lastResult.gravitationalPE;
        line(QString("Total  E: %1 J").arg(totalE, 0, 'e', 3));

        // Beam info
        y += 10;
        p.setPen(QColor(140, 150, 170));
        QFont f2("Sans", 9);
        p.setFont(f2);
        line(QString("Beam: %1×%2×%3 m").arg(BEAM_Lx).arg(BEAM_Ly).arg(BEAM_Lz));
        line(QString("E = %1 Pa,  ν = %2,  ρ = %3")
             .arg(MAT_E, 0, 'e', 1).arg(MAT_NU).arg(MAT_RHO));

        // Controls (right side)
        p.setPen(QColor(120, 120, 140));
        int bx = w - 280, by2 = 24;
        auto rline = [&](const QString& s) { p.drawText(bx, by2, s); by2 += 18; };
        rline("Sol sürükle    Döndür");
        rline("Sağ sürükle    Kaydır");
        rline("Tekerlek       Yakınlaştır");
        rline("W / S          ω  Artır / Azalt");
        rline("G              Gravity aç/kapat");
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
    QApplication app(argc, argv);
    SpinningBeamWidget win;
    win.show();
    return app.exec();
}
