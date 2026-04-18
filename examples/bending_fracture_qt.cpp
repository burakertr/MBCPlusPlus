/**
 * bending_fracture_qt.cpp
 * ═══════════════════════════════════════════════════════════════
 * Bending Fracture Test — Clamped-Clamped Beam with Notch + Cylinder
 *
 * A cylinder sits at the midpoint of a clamped-clamped beam (with
 * a V-notch at center) and pushes down with a prescribed Y
 * displacement profile.  Steel beam uses von Mises yield criterion
 * — elements whose centroid stress exceeds σ_yield are deleted.
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
#include <iostream>

#include "mb/fem/FlexibleBody.h"
#include "mb/fem/FlexibleIntegrators.h"
#include "mb/fem/FlexibleContactManager.h"
#include "mb/core/ThreadConfig.h"

using namespace mb;

// ═══════════════════════════════════════════════════════════════
//  Parameters
// ═══════════════════════════════════════════════════════════════

// Beam (steel, clamped-clamped, with center notch)
static constexpr double BEAM_LX  = 1.0;      // length (m)
static constexpr double BEAM_LY  = 0.04;     // height (m)
static constexpr double BEAM_LZ  = 0.06;     // depth  (m)
static constexpr int    BEAM_NX  = 14;
static constexpr int    BEAM_NY  = 3;
static constexpr int    BEAM_NZ  = 3;
static constexpr double BEAM_E   = 2e5;      // Pa (scaled for visible deformation)
static constexpr double BEAM_NU  = 0.3;
static constexpr double BEAM_RHO = 70.0;   // kg/m³

// Yield stress (scaled to match E scale)
static constexpr double YIELD_STRESS = 5e3;   // Pa

// Cylinder (indenter — very stiff)
static constexpr double CYL_R    = 0.025;    // radius (m)
static constexpr double CYL_L    = 0.05;     // length along Z (m)
static constexpr int    CYL_NR   = 2;
static constexpr int    CYL_NT   = 8;
static constexpr int    CYL_NZ   = 2;
static constexpr double CYL_E    = 1e7;      // Pa (very stiff)
static constexpr double CYL_NU   = 0.3;
static constexpr double CYL_RHO  = 70.0;

// Notch parameters
static constexpr double NOTCH_WIDTH = 0.02;  // notch width in X (m)
static constexpr double NOTCH_DEPTH = 0.02;  // notch depth from top in Y (m)

// Displacement profile
static constexpr double DISP_RATE = 0.04;    // m/s downward
static constexpr double MAX_DISP  = 0.15;    // max displacement (m)

// Simulation
static constexpr int    NSUB     = 80;

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
    int totalFractured = 0;
    double maxVonMises = 0;
    double currentDisp = 0;     // Current prescribed displacement

    // Per-element stress ratio (σ_vM / σ_yield) for colouring
    std::vector<double> elemStressRatio;   // indexed by all elements
    std::vector<double> aliveStressRatio;  // indexed by alive elements only (matches tetBeam)

    // History for graphs
    struct Sample {
        double time;
        double midDefl;     // mm
        double maxStress;   // MPa
        int fractured;
    };
    std::deque<Sample> history;

    void build() {
        // ── Contact config ──
        FlexContactConfig cfg;
        cfg.hertzExponent = 1.5;
        cfg.maxStiffness = 1e6;
        contactMgr = std::make_unique<FlexibleContactManager>(cfg);

        // ── Beam: clamped-clamped with center notch ──
        {
            auto mesh = generateBoxTetMesh(BEAM_LX, BEAM_LY, BEAM_LZ,
                                           BEAM_NX, BEAM_NY, BEAM_NZ);
            for (auto& n : mesh.nodes)
                n.z -= BEAM_LZ * 0.5;

            // Remove mesh elements in notch region before building FE body
            // Notch: centered at x=L/2, width=NOTCH_WIDTH, from top down NOTCH_DEPTH
            double notchXmin = BEAM_LX * 0.5 - NOTCH_WIDTH * 0.5;
            double notchXmax = BEAM_LX * 0.5 + NOTCH_WIDTH * 0.5;
            double notchYmin = BEAM_LY - NOTCH_DEPTH;  // top of beam minus depth
            {
                std::vector<GmshElement> filtered;
                for (auto& elem : mesh.elements) {
                    // Compute element centroid
                    double cx = 0, cy = 0;
                    for (int ni = 0; ni < 4; ni++) {
                        int nid = elem.nodeIds[ni];
                        // Find node by id
                        for (auto& nd : mesh.nodes) {
                            if (nd.id == nid) { cx += nd.x; cy += nd.y; break; }
                        }
                    }
                    cx /= 4; cy /= 4;
                    // Keep element if NOT in notch region
                    bool inNotch = (cx >= notchXmin && cx <= notchXmax && cy >= notchYmin);
                    if (!inNotch) filtered.push_back(elem);
                }
                mesh.elements = filtered;
            }

            ElasticMaterialProps mat{BEAM_E, BEAM_NU, BEAM_RHO, MaterialType::StVenantKirchhoff};
            beam = FlexibleBody::fromMesh(mesh, mat, "Beam", true);
            beam->gravity = {0, -9.81, 0};
            beam->dampingAlpha = 20.0;

            // Clamped ends (all DOFs fixed)
            beam->fixNodesOnPlane('x', 0.0, 1e-4);
            beam->fixNodesOnPlane('x', BEAM_LX, 1e-4);

            intBeam = std::make_unique<ImplicitFlexIntegrator>(*beam);
            intBeam->hhtAlpha = -0.15;
            intBeam->newtonTol = 1e-4;
            intBeam->maxNewtonIter = 30;
        }

        // ── Cylinder (rigid indenter) ──
        {
            auto mesh = generateCylinderTetMesh(CYL_R, CYL_L,
                                                 CYL_NR, CYL_NT, CYL_NZ);
            double gap = 0.01;  // enough gap to avoid premature contact
            for (auto& n : mesh.nodes) {
                n.x += BEAM_LX * 0.5;            // center of beam
                n.y += BEAM_LY + CYL_R + gap;    // on top of beam
                n.z -= CYL_L * 0.5;
            }

            ElasticMaterialProps mat{CYL_E, CYL_NU, CYL_RHO, MaterialType::NeoHookean};
            cyl = FlexibleBody::fromMesh(mesh, mat, "Cylinder", true);
            cyl->gravity = {0, -9.81, 0};
            cyl->dampingAlpha = 10.0;

            intCyl = std::make_unique<ImplicitFlexIntegrator>(*cyl);
            intCyl->hhtAlpha = -0.15;
            intCyl->newtonTol = 1e-4;
            intCyl->maxNewtonIter = 30;
        }

        tetBeam = beam->getTetConnectivity();
        tetCyl  = cyl->getTetConnectivity();

        FlexContactMaterial matBeam{0.5, 0.3, BEAM_E, BEAM_NU};
        FlexContactMaterial matCyl {0.5, 0.3, CYL_E,  CYL_NU};
        contactMgr->addBody(*beam, matBeam);
        contactMgr->addBody(*cyl,  matCyl);
        contactMgr->setContactMargin(0.001);   // tight margin to avoid premature contact
        contactMgr->setMaxDepth(0.005);
        contactMgr->maxForcePerDof = 1000.0;

        time = 0;
        totalFractured = 0;
        maxVonMises = 0;
        currentDisp = 0;
        history.clear();
        updateRenderData();
        computeStressField();
    }

    void step(double dt) {
        double ds = dt / NSUB;
        Vec3 g = gravityOn ? Vec3{0, -9.81, 0} : Vec3{0, 0, 0};
        beam->gravity = g;
        cyl->gravity = g;

        for (int s = 0; s < NSUB; s++) {
            // ── Prescribed displacement for cylinder ──
            double targetDisp = std::min(DISP_RATE * (time + s * ds), MAX_DISP);
            applyPrescribedDisplacement(targetDisp);

            // ── Contact ──
            contactMgr->step();
            auto fBeam = contactMgr->getContactForces(beam->id);
            auto fCyl  = contactMgr->getContactForces(cyl->id);

            beam->externalForces = !fBeam.empty()
                ? [fBeam](FlexibleBody&) { return fBeam; }
                : std::function<std::vector<double>(FlexibleBody&)>(nullptr);
            cyl->externalForces = !fCyl.empty()
                ? [fCyl](FlexibleBody&) { return fCyl; }
                : std::function<std::vector<double>(FlexibleBody&)>(nullptr);

            // ── Integrate ──
            intBeam->step(ds);
            // Cylinder: not integrated (prescribed motion)

            // ── Velocity clamp for beam ──
            {
                auto qd = beam->getFlexQd(); bool clamped = false;
                int nn = beam->nodes.size();
                for (int i = 0; i < nn; i++) {
                    double vx = qd[i*12], vy = qd[i*12+1], vz = qd[i*12+2];
                    double vm = std::sqrt(vx*vx + vy*vy + vz*vz);
                    if (vm > 5.0) {
                        double sc = 5.0 / vm;
                        for (int d = 0; d < 12; d++) qd[i*12+d] *= sc;
                        clamped = true;
                    }
                }
                if (clamped) beam->setFlexQd(qd);
            }
        }

        time += dt;
        currentDisp = std::min(DISP_RATE * time, MAX_DISP);
        updateRenderData();
        computeStressField();

        // ── Fracture check ──
        checkFracture();

        // ── Record history ──
        recordHistory();
    }

    void applyPrescribedDisplacement(double disp) {
        // Move all cylinder nodes to a uniform Y offset from their reference
        for (auto& nd : cyl->nodes) {
            nd.q[1] = nd.X0[1] - disp;
            nd.qd[1] = -DISP_RATE;  // velocity
        }
    }

    void computeStressField() {
        int nElem = (int)beam->elements.size();
        elemStressRatio.resize(nElem);
        maxVonMises = 0;
        for (int e = 0; e < nElem; e++) {
            double vm = beam->computeElementVonMises(e);
            elemStressRatio[e] = vm / YIELD_STRESS;
            maxVonMises = std::max(maxVonMises, vm);
        }

        // Build alive-only stress ratios (matches tetBeam from getTetConnectivity)
        aliveStressRatio.clear();
        for (int e = 0; e < nElem; e++) {
            if (beam->elements[e].alive)
                aliveStressRatio.push_back(elemStressRatio[e]);
        }
    }

    void checkFracture() {
        int nElem = (int)beam->elements.size();
        std::vector<int> failed;
        for (int e = 0; e < nElem; e++) {
            if (beam->elements[e].alive && elemStressRatio[e] >= 1.0)
                failed.push_back(e);
        }
        if (!failed.empty()) {
            totalFractured += (int)failed.size();
            beam->removeElements(failed);
            tetBeam = beam->getTetConnectivity();
            updateRenderData();
            computeStressField();

            // Invalidate integrator's cached mass inverse
            intBeam = std::make_unique<ImplicitFlexIntegrator>(*beam);
            intBeam->hhtAlpha = -0.15;
            intBeam->newtonTol = 1e-4;
            intBeam->maxNewtonIter = 30;

            // Invalidate contact manager's surface cache
            contactMgr->invalidateCache(beam->id);

            std::cout << "[FRACTURE] t=" << time << "s  removed "
                      << failed.size() << " elements (total: "
                      << totalFractured << ")\n";
        }
    }

    void recordHistory() {
        double midX = BEAM_LX * 0.5;
        double bestDist = 1e10, midY = BEAM_LY * 0.5;
        for (auto& p : posBeam) {
            double dist = std::abs(p.x - midX);
            if (dist < bestDist) { bestDist = dist; midY = p.y; }
        }
        double defl = (BEAM_LY * 0.5 - midY) * 1e3;  // mm

        history.push_back({time, defl, maxVonMises * 1e-6, totalFractured});
        if (history.size() > 500) history.pop_front();
    }

    void updateRenderData() {
        posBeam = beam->getNodePositions();
        posCyl  = cyl->getNodePositions();
    }

    int numContacts() const { return contactMgr ? contactMgr->numContacts() : 0; }
    Vec3 cylinderCentroid() const { return bodyCentroid(posCyl); }
};

// ═══════════════════════════════════════════════════════════════
//  Colour map: blue → cyan → green → yellow → red
// ═══════════════════════════════════════════════════════════════
static QColor stressColor(double ratio) {
    ratio = std::clamp(ratio, 0.0, 1.0);
    double r, g, b;
    if (ratio < 0.25)      { double s = ratio/0.25;        r=0; g=s; b=1; }
    else if (ratio < 0.5)  { double s = (ratio-0.25)/0.25; r=0; g=1; b=1-s; }
    else if (ratio < 0.75) { double s = (ratio-0.5)/0.25;  r=s; g=1; b=0; }
    else                   { double s = (ratio-0.75)/0.25; r=1; g=1-s; b=0; }
    return QColor(int(r*255), int(g*255), int(b*255));
}

// ═══════════════════════════════════════════════════════════════
//  Qt Widget
// ═══════════════════════════════════════════════════════════════
class FractureWidget : public QWidget {
public:
    FractureWidget(QWidget* parent = nullptr) : QWidget(parent) {
        setWindowTitle("MBC++ — Bending Fracture Test (Clamped + Notch)");
        resize(1400, 850);
        setMinimumSize(1000, 600);

        sim_.build();

        timer_ = new QTimer(this);
        connect(timer_, &QTimer::timeout, this, &FractureWidget::tick);
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

        p.fillRect(rect(), QColor(10, 12, 20));

        double scale = zoom_ * std::min(w, h) * 0.50 / BEAM_LX;
        double cx = w * 0.38 + panX_;
        double cy = h * 0.48 + panY_;

        double ca = std::cos(azimuth_), sa = std::sin(azimuth_);
        double cEl = std::cos(elevation_), sEl = std::sin(elevation_);

        auto toScreen = [&](Vec3 v) -> QPointF {
            double x1 = v.x * ca + v.z * sa;
            double y1 = v.y;
            double z1 = -v.x * sa + v.z * ca;
            return {cx + x1 * scale, cy - (y1 * cEl - z1 * sEl) * scale};
        };

        // ── Grid ──
        p.setPen(QPen(QColor(25, 28, 38), 1));
        double step = 0.05;
        for (double m = -0.2; m <= BEAM_LX + 0.2 + 1e-6; m += step)
            p.drawLine(toScreen({m, -0.08, -0.08}), toScreen({m, -0.08, 0.08}));
        for (double m = -0.08; m <= 0.08 + 1e-6; m += step)
            p.drawLine(toScreen({-0.2, -0.08, m}), toScreen({BEAM_LX + 0.2, -0.08, m}));

        // ── Axes ──
        {
            auto drawAxis = [&](Vec3 tip, QColor c, const QString& l) {
                p.setPen(QPen(c, 2));
                p.drawLine(toScreen({0,0,0}), toScreen(tip));
                p.setFont(QFont("Sans", 9, QFont::Bold));
                p.drawText(toScreen(tip) + QPointF(4, -4), l);
            };
            drawAxis({0.10,0,0}, QColor(220,70,70),  "X");
            drawAxis({0,0.10,0}, QColor(70,200,70),  "Y");
            drawAxis({0,0,0.10}, QColor(70,130,255), "Z");
        }

        // ── Pin supports ──
        {
            p.setPen(QPen(QColor(200, 200, 200), 2));
            auto drawPin = [&](double xPos) {
                Vec3 base(xPos, 0, 0);
                Vec3 left(xPos - 0.02, -0.025, 0);
                Vec3 right(xPos + 0.02, -0.025, 0);
                p.drawLine(toScreen(base), toScreen(left));
                p.drawLine(toScreen(base), toScreen(right));
                p.drawLine(toScreen(left), toScreen(right));
                for (int i = -2; i <= 2; i++) {
                    double xx = xPos + i * 0.008;
                    p.drawLine(toScreen({xx, -0.025, 0}),
                               toScreen({xx - 0.01, -0.035, 0}));
                }
            };
            drawPin(0.0);
            drawPin(BEAM_LX);
        }

        // ── Draw beam with stress colouring ──
        drawBeamStress(p, toScreen);

        // ── Draw cylinder ──
        drawCylinder(p, toScreen);

        // ── Contact points ──
        if (sim_.contactMgr && sim_.contactMgr->hasContacts()) {
            p.setPen(Qt::NoPen);
            p.setBrush(QColor(255, 255, 0, 220));
            for (auto& c : sim_.contactMgr->activeContacts()) {
                QPointF pt = toScreen(c.point);
                p.drawEllipse(pt, 3, 3);
            }
        }

        // ── Colour bar ──
        drawColourBar(p, w, h);

        // ── Graphs ──
        drawGraphs(p, w, h);

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

    // ── Helper: screen-space transform functor type ──
    template<typename F>
    void drawBeamStress(QPainter& p, const F& toScreen) {
        auto& pos = sim_.posBeam;
        auto& tets = sim_.tetBeam;
        auto& sr = sim_.aliveStressRatio;
        int nElem = (int)tets.size();

        // Face sort
        static const int fi[4][3] = {{1,2,3},{0,2,3},{0,1,3},{0,1,2}};
        struct Face { QPointF pts[3]; double depth; QColor col; };
        std::vector<Face> faces;
        faces.reserve(nElem * 4);

        double sa = std::sin(azimuth_);
        double ca_loc = std::cos(azimuth_);

        for (int e = 0; e < nElem; e++) {
            QColor col = stressColor(std::min(sr[e], 1.0));
            col.setAlpha(170);

            QPointF sp[4];
            for (int k = 0; k < 4; k++) sp[k] = toScreen(pos[tets[e][k]]);

            for (int f = 0; f < 4; f++) {
                Face fc;
                fc.col = col;
                fc.depth = 0;
                for (int k = 0; k < 3; k++) {
                    fc.pts[k] = sp[fi[f][k]];
                    auto& v = pos[tets[e][fi[f][k]]];
                    fc.depth += (-v.x * sa + v.z * ca_loc);
                }
                fc.depth /= 3;
                faces.push_back(fc);
            }
        }
        std::sort(faces.begin(), faces.end(),
                  [](const Face& a, const Face& b) { return a.depth < b.depth; });

        for (auto& f : faces) {
            QPainterPath path;
            path.moveTo(f.pts[0]); path.lineTo(f.pts[1]);
            path.lineTo(f.pts[2]); path.closeSubpath();
            p.fillPath(path, f.col);
        }

        // Edges (subtle)
        std::set<std::pair<int,int>> edgeSet;
        for (auto& t : tets) {
            for (int a = 0; a < 4; a++)
                for (int b = a+1; b < 4; b++)
                    edgeSet.insert({std::min(t[a],t[b]), std::max(t[a],t[b])});
        }
        p.setPen(QPen(QColor(255, 255, 255, 60), 0.5));
        for (auto& [a, b] : edgeSet)
            p.drawLine(toScreen(pos[a]), toScreen(pos[b]));
    }

    template<typename F>
    void drawCylinder(QPainter& p, const F& toScreen) {
        auto& pos = sim_.posCyl;
        auto& tets = sim_.tetCyl;

        static const int fi[4][3] = {{1,2,3},{0,2,3},{0,1,3},{0,1,2}};
        struct Face { QPointF pts[3]; double depth; };
        std::vector<Face> faces;
        faces.reserve(tets.size() * 4);

        double sa = std::sin(azimuth_);
        double ca_loc = std::cos(azimuth_);

        for (auto& t : tets) {
            QPointF sp[4];
            for (int k = 0; k < 4; k++) sp[k] = toScreen(pos[t[k]]);
            for (int f = 0; f < 4; f++) {
                Face fc;
                fc.depth = 0;
                for (int k = 0; k < 3; k++) {
                    fc.pts[k] = sp[fi[f][k]];
                    auto& v = pos[t[fi[f][k]]];
                    fc.depth += (-v.x * sa + v.z * ca_loc);
                }
                fc.depth /= 3;
                faces.push_back(fc);
            }
        }
        std::sort(faces.begin(), faces.end(),
                  [](const Face& a, const Face& b) { return a.depth < b.depth; });

        QColor cylCol(240, 160, 50, 160);
        for (auto& f : faces) {
            QPainterPath path;
            path.moveTo(f.pts[0]); path.lineTo(f.pts[1]);
            path.lineTo(f.pts[2]); path.closeSubpath();
            p.fillPath(path, cylCol);
        }

        // Edges
        std::set<std::pair<int,int>> edgeSet;
        for (auto& t : tets) {
            for (int a = 0; a < 4; a++)
                for (int b = a+1; b < 4; b++)
                    edgeSet.insert({std::min(t[a],t[b]), std::max(t[a],t[b])});
        }
        p.setPen(QPen(QColor(255, 200, 100, 140), 1.0));
        for (auto& [a, b] : edgeSet)
            p.drawLine(toScreen(pos[a]), toScreen(pos[b]));
    }

    void drawColourBar(QPainter& p, int w, int h) {
        int bx = w - 52, by = 60, bw = 18, bh = 200;
        for (int i = 0; i < bh; i++) {
            double t = 1.0 - double(i) / bh;
            p.setPen(stressColor(t));
            p.drawLine(bx, by + i, bx + bw, by + i);
        }
        p.setPen(QColor(140, 140, 160));
        p.drawRect(bx - 1, by - 1, bw + 2, bh + 2);

        QFont f("Sans", 8);
        p.setFont(f);
        p.drawText(bx - 5, by - 8,  "σ_yield");
        p.drawText(bx - 5,  by + bh + 14, "0");

        p.save();
        p.translate(bx + bw + 16, by + bh/2);
        p.rotate(-90);
        p.drawText(0, 0, "σ_vM / σ_yield");
        p.restore();
    }

    void drawGraphs(QPainter& p, int w, int h) {
        if (sim_.history.empty()) return;

        // ── Deflection graph ──
        {
            int gx = w - 400, gy = h - 200, gw = 370, gh = 80;
            p.fillRect(gx, gy, gw, gh, QColor(18, 20, 28, 210));
            p.setPen(QPen(QColor(50, 55, 70), 1));
            p.drawRect(gx, gy, gw, gh);

            p.setPen(QColor(160, 160, 180));
            QFont ft("Sans", 8, QFont::Bold);
            p.setFont(ft);
            p.drawText(gx + 10, gy + 13, "Orta Nokta Sehimi (mm)");

            int px = gx + 40, py = gy + 20, pw = gw - 55, ph = gh - 30;
            double maxT = sim_.time, minT = sim_.history.front().time;
            if (maxT - minT < 0.01) maxT = minT + 0.01;
            double maxD = 1.0;
            for (auto& s : sim_.history) maxD = std::max(maxD, std::abs(s.midDefl));

            p.setPen(QPen(QColor(80, 220, 120), 1.5));
            QPointF prev;
            bool first = true;
            for (auto& s : sim_.history) {
                double fx = (s.time - minT) / (maxT - minT);
                double fy = std::clamp(s.midDefl / maxD, 0.0, 1.0);
                QPointF pt(px + fx * pw, py + ph - fy * ph);
                if (!first) p.drawLine(prev, pt);
                prev = pt; first = false;
            }

            // Labels
            p.setPen(QColor(100, 100, 120));
            QFont fl("Sans", 7); p.setFont(fl);
            p.drawText(px - 35, py + 4, QString("%1").arg(maxD, 0, 'f', 1));
            p.drawText(px - 20, py + ph + 4, "0");
        }

        // ── Stress graph ──
        {
            int gx = w - 400, gy = h - 108, gw = 370, gh = 80;
            p.fillRect(gx, gy, gw, gh, QColor(18, 20, 28, 210));
            p.setPen(QPen(QColor(50, 55, 70), 1));
            p.drawRect(gx, gy, gw, gh);

            p.setPen(QColor(160, 160, 180));
            QFont ft("Sans", 8, QFont::Bold);
            p.setFont(ft);
            p.drawText(gx + 10, gy + 13, "Max σ_vM (MPa)");

            int px = gx + 40, py = gy + 20, pw = gw - 55, ph = gh - 30;
            double maxT = sim_.time, minT = sim_.history.front().time;
            if (maxT - minT < 0.01) maxT = minT + 0.01;
            double maxS = YIELD_STRESS * 1e-6;  // at least yield
            for (auto& s : sim_.history) maxS = std::max(maxS, s.maxStress);

            // Yield line
            double yieldY = std::clamp(YIELD_STRESS * 1e-6 / maxS, 0.0, 1.0);
            p.setPen(QPen(QColor(255, 60, 60, 150), 1, Qt::DashLine));
            int yy = py + ph - (int)(yieldY * ph);
            p.drawLine(px, yy, px + pw, yy);
            p.setPen(QColor(255, 80, 80));
            QFont fl("Sans", 7); p.setFont(fl);
            p.drawText(px + pw + 4, yy + 3, "yield");

            p.setPen(QPen(QColor(255, 160, 60), 1.5));
            QPointF prev;
            bool first = true;
            for (auto& s : sim_.history) {
                double fx = (s.time - minT) / (maxT - minT);
                double fy = std::clamp(s.maxStress / maxS, 0.0, 1.0);
                QPointF pt(px + fx * pw, py + ph - fy * ph);
                if (!first) p.drawLine(prev, pt);
                prev = pt; first = false;
            }

            p.setPen(QColor(100, 100, 120));
            p.drawText(px - 35, py + 4, QString("%1").arg(maxS, 0, 'f', 1));
            p.drawText(px - 20, py + ph + 4, "0");
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
        line(QString("Substeps: %1  Threads: %2").arg(NSUB).arg(ThreadConfig::numThreads()));
        line("");

        // Contact
        if (nc > 0) {
            p.setPen(QColor(255, 200, 50));
            QFont fb("Monospace", 11, QFont::Bold); p.setFont(fb);
            line(QString("● TEMAS  (%1 nokta)").arg(nc));
            p.setFont(f); p.setPen(QColor(200, 200, 200));
        } else {
            p.setPen(QColor(80, 200, 80));
            line("○ temas yok");
            p.setPen(QColor(200, 200, 200));
        }
        line("");

        // Stress info
        line(QString("σ_max: %1 MPa").arg(sim_.maxVonMises * 1e-6, 0, 'f', 2));
        line(QString("σ_yield: %1 MPa").arg(YIELD_STRESS * 1e-6, 0, 'f', 2));
        line(QString("σ/σ_y: %1")
             .arg(sim_.maxVonMises / YIELD_STRESS, 0, 'f', 3));
        line("");

        // Fracture info
        if (sim_.totalFractured > 0) {
            p.setPen(QColor(255, 60, 60));
            QFont fb("Monospace", 12, QFont::Bold); p.setFont(fb);
            line(QString("⚠ KIRILMA: %1 eleman").arg(sim_.totalFractured));
            p.setFont(f); p.setPen(QColor(200, 200, 200));
        } else {
            p.setPen(QColor(80, 200, 80));
            line("✓ Kırılma yok");
            p.setPen(QColor(200, 200, 200));
        }
        line("");

        // Displacement
        line(QString("Yer değ.: %1 mm").arg(sim_.currentDisp * 1e3, 0, 'f', 2));
        int aliveCount = 0;
        for (auto& e : sim_.beam->elements) if (e.alive) aliveCount++;
        line(QString("Kalan elem: %1").arg(aliveCount));
        line("");

        // Body info
        p.setPen(QColor(130, 135, 160));
        QFont f2("Sans", 9); p.setFont(f2);
        line(QString("Çubuk: %1×%2×%3 m  E=%4 Pa")
             .arg(BEAM_LX).arg(BEAM_LY).arg(BEAM_LZ).arg(BEAM_E, 0, 'e', 1));
        line(QString("Silindir: R=%1 L=%2 m")
             .arg(CYL_R).arg(CYL_L));

        // Controls
        p.setPen(QColor(100, 105, 130));
        int bx = w - 280, by = 24;
        auto rline = [&](const QString& s) { p.drawText(bx, by, s); by += 18; };
        rline("Sol sürükle    Döndür");
        rline("Sağ sürükle    Kaydır");
        rline("Tekerlek       Yakınlaştır");
        rline("SPACE          Duraklat");
        rline("R              Sıfırla");
        rline("G              Yerçekimi");
        rline("ESC            Çıkış");

        p.setPen(stressColor(0.0));
        p.drawText(bx, by + 8, "■ Çubuk (clamped, çentikli)");
        p.setPen(QColor(240, 160, 50));
        p.drawText(bx, by + 26, "■ Silindir (turuncu, prescribed)");

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
    printf("[MBC++] Bending Fracture Test\n");
    printf("[MBC++] Yield stress: %.2f MPa\n", YIELD_STRESS * 1e-6);

    QApplication app(argc, argv);
    FractureWidget win;
    win.show();
    return app.exec();
}
