/**
 * rolling_cylinder_qt.cpp
 * ═══════════════════════════════════════════════════════════════
 * Pinned-Pinned Elastik Çubuk Üzerinde İlerleyen Silindir
 *
 * Beam  : ANCF tet mesh, iki ucu sabit (x=0 ve x=L pinned)
 * Cylinder: ANCF tet mesh, başlangıç hızı ile çubuk üzerinde ilerler
 * Contact: Hertz + Flores contact model (FlexibleContactManager)
 *
 * Klasik "moving load on a simply-supported beam" probleminin
 * sonlu-eleman tabanlı, temas içeren tam dinamik analizi.
 *
 * Controls:
 *   Space   — Pause / Resume
 *   R       — Reset
 *   G       — Toggle gravity
 *   +/-     — Zoom
 *   Left drag  — Orbit
 *   Right drag — Pan
 *   Wheel      — Zoom
 *   ESC        — Quit
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
#include <set>
#include <memory>
#include <deque>

#include "mb/fem/FlexibleBody.h"
#include "mb/fem/FlexibleIntegrators.h"
#include "mb/fem/FlexibleContactManager.h"
#include "mb/core/ThreadConfig.h"
#include "mb/fem/MeshGenerators.h"

using namespace mb;

// ═══════════════════════════════════════════════════════════════
//  Parameters
// ═══════════════════════════════════════════════════════════════

// Beam (pinned-pinned): uzun, ince çelik çubuk
static constexpr double BEAM_LX  = 1.0;     // length (m)
static constexpr double BEAM_LY  = 0.04;    // height (m)
static constexpr double BEAM_LZ  = 0.06;    // depth  (m)
static constexpr int    BEAM_NX  = 12;
static constexpr int    BEAM_NY  = 2;
static constexpr int    BEAM_NZ  = 2;
static constexpr double BEAM_E   = 2e7;     // Pa (yumuşak, sehimi görmek için)
static constexpr double BEAM_NU  = 0.3;
static constexpr double BEAM_RHO = 7800.0;  // kg/m³

// Cylinder: yuvarlanan silindir
static constexpr double CYL_R    = 0.03;    // radius (m)
static constexpr double CYL_L    = 0.05;    // length along Z (m)
static constexpr int    CYL_NR   = 2;       // radial divisions
static constexpr int    CYL_NT   = 8;       // tangential divisions
static constexpr int    CYL_NZ   = 2;       // axial divisions
static constexpr double CYL_E    = 5e8;     // Pa (sert silindir)
static constexpr double CYL_NU   = 0.3;
static constexpr double CYL_RHO  = 7800.0;  // kg/m³

// Simulation
static constexpr int    NSUB     = 100;      // sub-steps per frame
static constexpr double CYL_V0   = 0.3;     // initial horizontal velocity (m/s)
static constexpr double CYL_X0   = 0.10;    // initial cylinder center X position

// ═══════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════
static Vec3 bodyCentroid(const std::vector<Vec3>& pos) {
    Vec3 c{};
    for (auto& p : pos) { c.x += p.x; c.y += p.y; c.z += p.z; }
    double n = pos.size();
    return {c.x / n, c.y / n, c.z / n};
}

// ═══════════════════════════════════════════════════════════════
//  Simulation
// ═══════════════════════════════════════════════════════════════
struct Sim {
    std::shared_ptr<FlexibleBody>           beam, cyl;
    std::unique_ptr<ImplicitFlexIntegrator> intBeam, intCyl;
    std::vector<std::array<int,4>>          tetBeam, tetCyl;
    std::unique_ptr<FlexibleContactManager> contactMgr;

    std::vector<Vec3> posBeam, posCyl;
    double time = 0;
    bool gravityOn = true;

    // Midpoint deflection history (for plot)
    struct DeflectionSample {
        double time;
        double cylX;       // cylinder centroid X
        double midDefl;    // beam midpoint Y deflection
    };
    std::deque<DeflectionSample> deflHistory;
    double maxDeflection = 0;

    void build() {
        // ── Contact config ──
        FlexContactConfig cfg;
        cfg.hertzExponent = 1.5;
        cfg.maxStiffness = 1e6;
        contactMgr = std::make_unique<FlexibleContactManager>(cfg);

        // ── Beam: pinned-pinned ──
        {
            auto mesh = mb::generateBoxTetMesh(BEAM_LX, BEAM_LY, BEAM_LZ,
                                           BEAM_NX, BEAM_NY, BEAM_NZ);
            // Center Z around 0, X starts at 0
            for (auto& n : mesh.nodes) {
                n.z -= BEAM_LZ * 0.5;
            }

            ElasticMaterialProps mat{BEAM_E, BEAM_NU, BEAM_RHO, MaterialType::NeoHookean};
            beam = FlexibleBody::fromMesh(mesh, mat, "Beam", true);
            beam->gravity = {0, -9.81, 0};
            beam->dampingAlpha = 15.0;  // moderate damping

            // Pin left end (x ≈ 0) — fix all DOFs
            beam->fixNodesOnPlane('x', 0.0, 1e-4);
            // Pin right end (x ≈ BEAM_LX) — fix all DOFs
            beam->fixNodesOnPlane('x', BEAM_LX, 1e-4);

            intBeam = std::make_unique<ImplicitFlexIntegrator>(*beam);
            intBeam->hhtAlpha = -0.1;
            intBeam->newtonTol = 1e-4;
            intBeam->maxNewtonIter = 25;
        }

        // ── Cylinder ──
        {
            auto mesh = generateCylinderTetMesh(CYL_R, CYL_L,
                                                 CYL_NR, CYL_NT, CYL_NZ);
            // Position cylinder: centered at (CYL_X0, BEAM_LY + CYL_R + gap, 0)
            // Cylinder mesh: default axis is Z, length CYL_L, centered at origin
            // We need to rotate it so axis is along Z, place it on top of beam
            double gap = 0.002;  // small gap so contact kicks in naturally
            for (auto& n : mesh.nodes) {
                n.x += CYL_X0;
                n.y += BEAM_LY + CYL_R + gap;
                n.z -= CYL_L * 0.5;  // center along Z
            }

            ElasticMaterialProps mat{CYL_E, CYL_NU, CYL_RHO, MaterialType::NeoHookean};
            cyl = FlexibleBody::fromMesh(mesh, mat, "Cylinder", true);
            cyl->gravity = {0, -9.81, 0};
            cyl->dampingAlpha = 5.0;

            // Initial velocity: uniform Vx
            cyl->setLinearVelocityFlex({CYL_V0, 0, 0});

            intCyl = std::make_unique<ImplicitFlexIntegrator>(*cyl);
            intCyl->hhtAlpha = -0.1;
            intCyl->newtonTol = 1e-4;
            intCyl->maxNewtonIter = 25;
        }

        tetBeam = beam->getTetConnectivity();
        tetCyl  = cyl->getTetConnectivity();

        // Register bodies with contact manager
        FlexContactMaterial matBeam{0.6, 0.3, BEAM_E, BEAM_NU};
        FlexContactMaterial matCyl {0.6, 0.3, CYL_E,  CYL_NU};
        contactMgr->addBody(*beam, matBeam);
        contactMgr->addBody(*cyl,  matCyl);
        contactMgr->setContactMargin(0.005);
        contactMgr->setMaxDepth(0.008);
        contactMgr->maxForcePerDof = 800.0;

        time = 0;
        deflHistory.clear();
        maxDeflection = 0;
        updateRenderData();
    }

    void step(double dt) {
        double ds = dt / NSUB;
        Vec3 g = gravityOn ? Vec3{0, -9.81, 0} : Vec3{0, 0, 0};
        beam->gravity = g;
        cyl->gravity  = g;

        for (int s = 0; s < NSUB; s++) {
            contactMgr->step();

            auto fBeam = contactMgr->getContactForces(beam->id);
            auto fCyl  = contactMgr->getContactForces(cyl->id);

            beam->externalForces = !fBeam.empty()
                ? [fBeam](FlexibleBody&) { return fBeam; }
                : std::function<std::vector<double>(FlexibleBody&)>(nullptr);
            cyl->externalForces = !fCyl.empty()
                ? [fCyl](FlexibleBody&) { return fCyl; }
                : std::function<std::vector<double>(FlexibleBody&)>(nullptr);

            intBeam->step(ds);
            intCyl->step(ds);

            // Velocity clamp
            auto clampVel = [](FlexibleBody& b, double vmax) {
                auto qd = b.getFlexQd(); bool clamped = false;
                int nn = b.nodes.size();
                for (int i = 0; i < nn; i++) {
                    double vx = qd[i*12], vy = qd[i*12+1], vz = qd[i*12+2];
                    double vm = std::sqrt(vx*vx + vy*vy + vz*vz);
                    if (vm > vmax) {
                        double sc = vmax / vm;
                        for (int d = 0; d < 12; d++) qd[i*12+d] *= sc;
                        clamped = true;
                    }
                }
                if (clamped) b.setFlexQd(qd);
            };
            clampVel(*beam, 5.0);
            clampVel(*cyl,  5.0);
        }

        time += dt;
        updateRenderData();

        // Record midpoint deflection
        recordDeflection();
    }

    void updateRenderData() {
        posBeam = beam->getNodePositions();
        posCyl  = cyl->getNodePositions();
    }

    void recordDeflection() {
        // Find beam midpoint node (closest to x=L/2)
        double midX = BEAM_LX * 0.5;
        double bestDist = 1e10;
        double midY = 0;
        for (auto& p : posBeam) {
            double dx = p.x - midX;
            double dist = std::abs(dx);
            if (dist < bestDist) {
                bestDist = dist;
                midY = p.y;
            }
        }
        double defl = BEAM_LY * 0.5 - midY;  // downward deflection (positive = down)
        Vec3 cylC = bodyCentroid(posCyl);

        deflHistory.push_back({time, cylC.x, defl});
        if (deflHistory.size() > 600) deflHistory.pop_front();

        maxDeflection = std::max(maxDeflection, std::abs(defl));
    }

    int numContacts() const { return contactMgr ? contactMgr->numContacts() : 0; }

    double kineticEnergy() const {
        double ke = 0;
        for (auto* b : {beam.get(), cyl.get()}) {
            auto qd = b->getFlexQd();
            auto md = b->getMassDiagonal();
            for (int i = 0; i < (int)qd.size(); i++)
                ke += 0.5 * md[i] * qd[i] * qd[i];
        }
        return ke;
    }

    Vec3 cylinderCentroid() const { return bodyCentroid(posCyl); }
};

// ═══════════════════════════════════════════════════════════════
//  Colour map: blue → cyan → green → yellow → red
// ═══════════════════════════════════════════════════════════════
static QColor heatmap(double t) {
    t = std::clamp(t, 0.0, 1.0);
    double r, g, b;
    if (t < 0.25)      { double s = t/0.25;        r=0; g=s; b=1; }
    else if (t < 0.5)  { double s = (t-0.25)/0.25; r=0; g=1; b=1-s; }
    else if (t < 0.75) { double s = (t-0.5)/0.25;  r=s; g=1; b=0; }
    else               { double s = (t-0.75)/0.25; r=1; g=1-s; b=0; }
    return QColor(int(r*255), int(g*255), int(b*255));
}

// ═══════════════════════════════════════════════════════════════
//  Qt Widget
// ═══════════════════════════════════════════════════════════════
class RollingCylinderWidget : public QWidget {
public:
    RollingCylinderWidget(QWidget* parent = nullptr) : QWidget(parent) {
        setWindowTitle("MBC++ — Pinned Çubuk Üzerinde İlerleyen Silindir");
        resize(1400, 800);
        setMinimumSize(1000, 600);

        sim_.build();

        timer_ = new QTimer(this);
        connect(timer_, &QTimer::timeout, this, &RollingCylinderWidget::tick);
        timer_->start(16);

        elapsed_.start();
        setFocusPolicy(Qt::StrongFocus);
        azimuth_ = 0.3;
        elevation_ = 0.25;
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        int w = width(), h = height();

        // Dark background
        p.fillRect(rect(), QColor(12, 14, 22));

        double scale = zoom_ * std::min(w, h) * 0.55 / BEAM_LX;
        double cx = w * 0.40 + panX_;
        double cy = h * 0.50 + panY_;

        double ca = std::cos(azimuth_), sa = std::sin(azimuth_);
        double cEl = std::cos(elevation_), sEl = std::sin(elevation_);

        auto toScreen = [&](Vec3 v) -> QPointF {
            double x1 = v.x * ca + v.z * sa;
            double y1 = v.y;
            double z1 = -v.x * sa + v.z * ca;
            return {cx + x1 * scale, cy - (y1 * cEl - z1 * sEl) * scale};
        };

        // ── Ground reference line ──
        p.setPen(QPen(QColor(28, 32, 44), 1));
        double step = 0.05;
        for (double m = -0.2; m <= BEAM_LX + 0.2 + 1e-6; m += step)
            p.drawLine(toScreen({m, -0.1, -0.1}), toScreen({m, -0.1, 0.1}));
        for (double m = -0.1; m <= 0.1 + 1e-6; m += step)
            p.drawLine(toScreen({-0.2, -0.1, m}), toScreen({BEAM_LX + 0.2, -0.1, m}));

        // ── Coordinate axes ──
        {
            auto drawAxis = [&](Vec3 tip, QColor c, const QString& l) {
                p.setPen(QPen(c, 2));
                p.drawLine(toScreen({0,0,0}), toScreen(tip));
                p.setFont(QFont("Sans", 9, QFont::Bold));
                p.drawText(toScreen(tip) + QPointF(4, -4), l);
            };
            drawAxis({0.12,0,0}, QColor(220,70,70),  "X");
            drawAxis({0,0.12,0}, QColor(70,200,70),  "Y");
            drawAxis({0,0,0.12}, QColor(70,130,255), "Z");
        }

        // ── Pin supports (triangle symbols at x=0 and x=L) ──
        {
            p.setPen(QPen(QColor(200, 200, 200), 2));
            auto drawPin = [&](double xPos) {
                Vec3 base(xPos, 0, 0);
                Vec3 left(xPos - 0.02, -0.025, 0);
                Vec3 right(xPos + 0.02, -0.025, 0);
                p.drawLine(toScreen(base), toScreen(left));
                p.drawLine(toScreen(base), toScreen(right));
                p.drawLine(toScreen(left), toScreen(right));
                // Ground hatch
                for (int i = -2; i <= 2; i++) {
                    double xx = xPos + i * 0.008;
                    Vec3 a(xx, -0.025, 0);
                    Vec3 b(xx - 0.01, -0.035, 0);
                    p.drawLine(toScreen(a), toScreen(b));
                }
            };
            drawPin(0.0);
            drawPin(BEAM_LX);
        }

        // ── Draw body helper ──
        auto drawBody = [&](const std::vector<Vec3>& pos,
                           const std::vector<std::array<int,4>>& tets,
                           QColor tint)
        {
            // Face sort for proper depth
            static const int fi[4][3] = {{1,2,3},{0,2,3},{0,1,3},{0,1,2}};
            struct Face { QPointF pts[3]; double depth; };
            std::vector<Face> faces;
            faces.reserve(tets.size() * 4);

            for (auto& t : tets) {
                QPointF sp[4];
                for (int k = 0; k < 4; k++) sp[k] = toScreen(pos[t[k]]);
                for (int f = 0; f < 4; f++) {
                    Face fc;
                    fc.depth = 0;
                    for (int k = 0; k < 3; k++) {
                        fc.pts[k] = sp[fi[f][k]];
                        auto& v = pos[t[fi[f][k]]];
                        fc.depth += (-v.x * sa + v.z * ca);
                    }
                    fc.depth /= 3;
                    faces.push_back(fc);
                }
            }
            std::sort(faces.begin(), faces.end(),
                      [](const Face& a, const Face& b) { return a.depth < b.depth; });

            QColor fc(tint.red(), tint.green(), tint.blue(), 150);
            for (auto& f : faces) {
                QPainterPath path;
                path.moveTo(f.pts[0]); path.lineTo(f.pts[1]);
                path.lineTo(f.pts[2]); path.closeSubpath();
                p.fillPath(path, fc);
            }

            // Edges
            std::set<std::pair<int,int>> edgeSet;
            for (auto& t : tets) {
                int n[4] = {t[0], t[1], t[2], t[3]};
                for (int a = 0; a < 4; a++)
                    for (int b = a+1; b < 4; b++)
                        edgeSet.insert({std::min(n[a],n[b]), std::max(n[a],n[b])});
            }
            p.setPen(QPen(QColor(255, 255, 255, 120), 0.8));
            for (auto& [a, b] : edgeSet)
                p.drawLine(toScreen(pos[a]), toScreen(pos[b]));
        };

        // ── Depth-sort and draw both bodies ──
        Vec3 cBeam = bodyCentroid(sim_.posBeam);
        Vec3 cCyl  = bodyCentroid(sim_.posCyl);
        double dBeam = -cBeam.x * sa + cBeam.z * ca;
        double dCyl  = -cCyl.x  * sa + cCyl.z  * ca;

        auto drawBeamBody = [&]() {
            drawBody(sim_.posBeam, sim_.tetBeam, QColor(60, 160, 220));  // mavi
        };
        auto drawCylBody = [&]() {
            drawBody(sim_.posCyl, sim_.tetCyl, QColor(240, 140, 40));   // turuncu
        };

        if (dBeam < dCyl) { drawBeamBody(); drawCylBody(); }
        else              { drawCylBody();  drawBeamBody(); }

        // ── Contact points ──
        if (sim_.contactMgr && sim_.contactMgr->hasContacts()) {
            p.setPen(Qt::NoPen);
            p.setBrush(QColor(255, 60, 60, 200));
            for (auto& c : sim_.contactMgr->activeContacts()) {
                QPointF pt = toScreen(c.point);
                p.drawEllipse(pt, 4, 4);
            }
        }

        // ── Draw deflection graph (bottom-right) ──
        drawDeflectionGraph(p, w, h);

        // ── HUD ──
        drawHUD(p, w, h);
    }

    void keyPressEvent(QKeyEvent* e) override {
        switch (e->key()) {
        case Qt::Key_Space: paused_ = !paused_; break;
        case Qt::Key_R:
            sim_.build(); elapsed_.restart(); frameCount_ = 0;
            azimuth_ = 0.3; elevation_ = 0.25; panX_ = panY_ = 0; zoom_ = 1.0;
            break;
        case Qt::Key_G: sim_.gravityOn = !sim_.gravityOn; break;
        case Qt::Key_Plus: case Qt::Key_Equal: zoom_ *= 1.2; break;
        case Qt::Key_Minus: zoom_ /= 1.2; break;
        case Qt::Key_Escape: close(); break;
        }
        update();
    }

    void mousePressEvent(QMouseEvent* e) override {
        lastMousePos_ = e->pos(); e->accept();
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
            panX_ += d.x(); panY_ += d.y();
        }
        update(); e->accept();
    }
    void wheelEvent(QWheelEvent* e) override {
        zoom_ *= (e->angleDelta().y() > 0) ? 1.15 : 1.0 / 1.15;
        update(); e->accept();
    }

private slots:
    void tick() {
        double dt = 1.0 / 60.0;
        if (!paused_) sim_.step(dt);
        frameCount_++;
        update();
    }

private:
    Sim sim_;
    QTimer* timer_ = nullptr;
    QElapsedTimer elapsed_;
    bool paused_ = false;
    int  frameCount_ = 0;
    double zoom_ = 1.0;
    double azimuth_ = 0, elevation_ = 0;
    double panX_ = 0, panY_ = 0;
    QPoint lastMousePos_;

    void drawDeflectionGraph(QPainter& p, int w, int h) {
        // Graph area (bottom-right)
        int gx = w - 380, gy = h - 200, gw = 350, gh = 170;

        // Background
        p.fillRect(gx, gy, gw, gh, QColor(20, 22, 30, 200));
        p.setPen(QPen(QColor(60, 65, 80), 1));
        p.drawRect(gx, gy, gw, gh);

        // Title
        p.setPen(QColor(180, 180, 200));
        QFont ft("Sans", 9, QFont::Bold);
        p.setFont(ft);
        p.drawText(gx + 10, gy + 16, "Orta Nokta Sehimi (mm)");

        if (sim_.deflHistory.empty()) return;

        // Plot area
        int px = gx + 45, py = gy + 28, pw = gw - 60, ph = gh - 50;

        // Find scale
        double maxT = sim_.time;
        double minT = sim_.deflHistory.front().time;
        if (maxT - minT < 0.01) maxT = minT + 0.01;

        double maxD = std::max(sim_.maxDeflection * 1e3, 0.1);  // mm

        // Grid
        p.setPen(QPen(QColor(40, 44, 55), 1));
        for (int i = 0; i <= 4; i++) {
            int yy = py + i * ph / 4;
            p.drawLine(px, yy, px + pw, yy);
        }

        // Axes labels
        p.setPen(QColor(120, 120, 140));
        QFont fl("Sans", 7);
        p.setFont(fl);
        p.drawText(px - 40, py + 4, QString("%1").arg(maxD, 0, 'f', 1));
        p.drawText(px - 40, py + ph + 4, "0.0");
        p.drawText(px - 5, py + ph + 14, QString("%1").arg(minT, 0, 'f', 1));
        p.drawText(px + pw - 20, py + ph + 14, QString("%1 s").arg(maxT, 0, 'f', 1));

        // Plot line
        p.setPen(QPen(QColor(80, 220, 120), 2));
        QPointF prev;
        bool first = true;
        for (auto& s : sim_.deflHistory) {
            double fx = (s.time - minT) / (maxT - minT);
            double fy = (s.midDefl * 1e3) / maxD;
            fy = std::clamp(fy, 0.0, 1.0);
            QPointF pt(px + fx * pw, py + ph - fy * ph);
            if (!first) p.drawLine(prev, pt);
            prev = pt;
            first = false;
        }

        // Current value
        if (!sim_.deflHistory.empty()) {
            double curDefl = sim_.deflHistory.back().midDefl * 1e3;
            p.setPen(QColor(80, 220, 120));
            QFont fv("Monospace", 9, QFont::Bold);
            p.setFont(fv);
            p.drawText(px + pw - 80, py - 2,
                       QString("%1 mm").arg(curDefl, 0, 'f', 2));
        }
    }

    void drawHUD(QPainter& p, int w, int /*h*/) {
        double fps = frameCount_ / (elapsed_.elapsed() * 0.001 + 1e-9);
        int nc = sim_.numContacts();

        p.setPen(QColor(200, 200, 200));
        QFont f("Monospace", 11);
        f.setStyleHint(QFont::Monospace);
        p.setFont(f);

        int x = 14, y = 24, dy = 20;
        auto line = [&](const QString& s) { p.drawText(x, y, s); y += dy; };

        line(QString("t = %1 s").arg(sim_.time, 0, 'f', 4));
        line(QString("FPS  %1").arg(fps, 0, 'f', 1));
        line(QString("Substeps: %1").arg(NSUB));
        line(QString("Threads: %1").arg(ThreadConfig::numThreads()));
        line("");

        // Contact info
        if (nc > 0) {
            p.setPen(QColor(255, 80, 80));
            QFont fb("Monospace", 12, QFont::Bold); p.setFont(fb);
            p.drawText(x, y, QString("● TEMAS  (%1 nokta)").arg(nc));
            y += dy + 2; p.setFont(f); p.setPen(QColor(200, 200, 200));
        } else {
            p.setPen(QColor(80, 200, 80));
            line("○ temas yok");
            p.setPen(QColor(200, 200, 200));
        }
        line("");

        // Cylinder position
        Vec3 cylC = sim_.cylinderCentroid();
        line(QString("Silindir X: %1 m").arg(cylC.x, 0, 'f', 4));
        line(QString("Silindir Y: %1 m").arg(cylC.y, 0, 'f', 4));
        line(QString("Max sehim: %1 mm").arg(sim_.maxDeflection * 1e3, 0, 'f', 2));
        line("");

        // Body info
        p.setPen(QColor(140, 150, 170));
        QFont f2("Sans", 9);
        p.setFont(f2);
        line(QString("Çubuk: %1×%2×%3 m  E=%4 Pa")
             .arg(BEAM_LX).arg(BEAM_LY).arg(BEAM_LZ).arg(BEAM_E, 0, 'e', 1));
        line(QString("Silindir: R=%1 L=%2 m  E=%3 Pa")
             .arg(CYL_R).arg(CYL_L).arg(CYL_E, 0, 'e', 1));
        line(QString("DOF çubuk: %1  silindir: %2")
             .arg(sim_.beam->numDof).arg(sim_.cyl->numDof));
        line(QString("KE: %1 J").arg(sim_.kineticEnergy(), 0, 'e', 3));
        line(QString("Gravity: %1").arg(sim_.gravityOn ? "ON" : "OFF"));

        // Controls (right side)
        p.setPen(QColor(120, 120, 140));
        int bx = w - 280, by = 24;
        auto rline = [&](const QString& s) { p.drawText(bx, by, s); by += 18; };
        rline("Sol sürükle    Döndür");
        rline("Sağ sürükle    Kaydır");
        rline("Tekerlek       Yakınlaştır");
        rline("SPACE          Duraklat");
        rline("R              Sıfırla");
        rline("G              Yerçekimi");
        rline("ESC            Çıkış");

        p.setPen(QColor(60, 160, 220));
        p.drawText(bx, by + 8, "■ Çubuk (mavi, pinned-pinned)");
        p.setPen(QColor(240, 140, 40));
        p.drawText(bx, by + 26, "■ Silindir (turuncu, ilerleyen)");

        if (paused_) {
            p.setPen(QColor(255, 80, 80));
            QFont fb("Sans", 16, QFont::Bold);
            p.setFont(fb);
            p.drawText(w / 2 - 50, 40, "⏸  DURAKLATILDI");
        }
    }
};

// ═══════════════════════════════════════════════════════════════
int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-c" && i + 1 < argc)
            ThreadConfig::setNumThreads(std::atoi(argv[++i]));
    }
    printf("[MBC++] Using %d OpenMP thread(s)\n", ThreadConfig::numThreads());

    QApplication app(argc, argv);
    RollingCylinderWidget win;
    win.show();
    return app.exec();
}
