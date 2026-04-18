/**
 * tet_collision2_qt.cpp
 * Two ANCF flexible BARS (tet-meshed) with contact via FlexibleContactManager.
 *
 * Body A : horizontal bar lying on the ground (fixed bottom)
 * Body B : vertical bar dropped from height
 *
 * Controls:  Space=Pause  R=Reset  G=Gravity  E=Nudge  ESC=Quit
 *            Left drag=Orbit  Right drag=Pan  Wheel=Zoom
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
#include <QImage>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <set>
#include <memory>

#include "mb/fem/FlexibleBody.h"
#include "mb/fem/FlexibleIntegrators.h"
#include "mb/fem/FlexibleContactManager.h"
#include "mb/core/ThreadConfig.h"

using namespace mb;

// ─── Parameters ──────────────────────────────────────────────────────────────
// Bar A: horizontal (lying along X)
static constexpr double BAR_A_LX = 0.40, BAR_A_LY = 0.06, BAR_A_LZ = 0.06;
static constexpr int    BAR_A_NX = 6,    BAR_A_NY = 2,     BAR_A_NZ = 2;
static constexpr double EA = 2e7,  RHO_A = 2700.0;   // aluminyum-ish

// Bar B: vertical (standing along Y, dropped)
static constexpr double BAR_B_LX = 0.05, BAR_B_LY = 0.30, BAR_B_LZ = 0.05;
static constexpr int    BAR_B_NX = 2,    BAR_B_NY = 5,     BAR_B_NZ = 2;
static constexpr double EB = 1e7,  RHO_B = 1800.0;   // plastik-ish

static constexpr double NU = 0.3;
static constexpr int    NSUB = 150;
static constexpr double DROP_HEIGHT = 0.35;   // B'nin alt yüzeyinin yerden yüksekliği

// ─── Helpers ─────────────────────────────────────────────────────────────────
static Vec3 bodyCentroid(const std::vector<Vec3>& pos) {
    Vec3 c{}; for (auto& p : pos) { c.x += p.x; c.y += p.y; c.z += p.z; }
    double n = pos.size(); return {c.x/n, c.y/n, c.z/n};
}

// ─── Simulation ──────────────────────────────────────────────────────────────
struct Sim {
    std::shared_ptr<FlexibleBody>           bodyA, bodyB;
    std::unique_ptr<ImplicitFlexIntegrator> intA,  intB;
    std::vector<std::array<int,4>>          tetA,  tetB;
    std::unique_ptr<FlexibleContactManager> contactMgr;
    std::vector<Vec3>   posA, posB;
    double time = 0;
    bool gravityOn = true;

    void build() {
        // Contact config
        FlexContactConfig cfg;
        cfg.hertzExponent = 1.5;
        cfg.maxStiffness = 5e5;
        contactMgr = std::make_unique<FlexibleContactManager>(cfg);

        // Body A: yatay çubuk, yerde (alt yüzey y=0)
        {
            auto mesh = generateBoxTetMesh(BAR_A_LX, BAR_A_LY, BAR_A_LZ,
                                           BAR_A_NX, BAR_A_NY, BAR_A_NZ);
            // Mesh orijin (0,0,0) — X boyunca yatay, ortala
            for (auto& n : mesh.nodes) { n.x -= BAR_A_LX*0.5; n.z -= BAR_A_LZ*0.5; }

            ElasticMaterialProps mat{EA, NU, RHO_A, MaterialType::NeoHookean};
            bodyA = FlexibleBody::fromMesh(mesh, mat, "BarA", true);
            bodyA->gravity = {0, -9.81, 0};
            bodyA->dampingAlpha = 8.0;
            bodyA->fixNodesOnPlane('y', 0.0, 1e-4);  // alt yüzey sabit
            intA = std::make_unique<ImplicitFlexIntegrator>(*bodyA);
            intA->hhtAlpha = -0.1;
            intA->newtonTol = 1e-4;
            intA->maxNewtonIter = 25;
        }

        // Body B: dikey çubuk, yüksekten bırakılıyor
        {
            auto mesh = generateBoxTetMesh(BAR_B_LX, BAR_B_LY, BAR_B_LZ,
                                           BAR_B_NX, BAR_B_NY, BAR_B_NZ);
            // Ortala ve DROP_HEIGHT yüksekliğe koy
            for (auto& n : mesh.nodes) {
                n.x -= BAR_B_LX*0.5;
                n.z -= BAR_B_LZ*0.5;
                n.y += DROP_HEIGHT;
            }

            ElasticMaterialProps mat{EB, NU, RHO_B, MaterialType::NeoHookean};
            bodyB = FlexibleBody::fromMesh(mesh, mat, "BarB", true);
            bodyB->gravity = {0, -9.81, 0};
            bodyB->dampingAlpha = 5.0;
            intB = std::make_unique<ImplicitFlexIntegrator>(*bodyB);
            intB->hhtAlpha = -0.1;
            intB->newtonTol = 1e-4;
            intB->maxNewtonIter = 25;
        }

        tetA = bodyA->getTetConnectivity();
        tetB = bodyB->getTetConnectivity();

        // Register with contact manager
        FlexContactMaterial matA{0.7, 0.4, EA, NU};
        FlexContactMaterial matB{0.7, 0.4, EB, NU};
        contactMgr->addBody(*bodyA, matA);
        contactMgr->addBody(*bodyB, matB);
        contactMgr->enableGround(GroundPlane{0.0, {0, 1, 0}});
        contactMgr->setContactMargin(0.004);
        contactMgr->setMaxDepth(0.008);
        contactMgr->maxForcePerDof = 500.0;

        time = 0;
        updateRenderData();
    }

    void step(double dt) {
        double ds = dt / NSUB;
        Vec3 g = gravityOn ? Vec3{0, -9.81, 0} : Vec3{0, 0, 0};
        bodyA->gravity = g;
        bodyB->gravity = g;

        for (int s = 0; s < NSUB; s++) {
            contactMgr->step();

            auto fA = contactMgr->getContactForces(bodyA->id);
            auto fB = contactMgr->getContactForces(bodyB->id);
            bodyA->externalForces = !fA.empty() ?
                [fA](FlexibleBody&) { return fA; } : std::function<std::vector<double>(FlexibleBody&)>(nullptr);
            bodyB->externalForces = !fB.empty() ?
                [fB](FlexibleBody&) { return fB; } : std::function<std::vector<double>(FlexibleBody&)>(nullptr);

            intA->step(ds);
            intB->step(ds);

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
            clampVel(*bodyA, 8.0);
            clampVel(*bodyB, 8.0);
        }
        time += dt;
        updateRenderData();
    }

    void updateRenderData() {
        posA = bodyA->getNodePositions();
        posB = bodyB->getNodePositions();
    }

    int numContacts() const { return contactMgr ? contactMgr->numContacts() : 0; }

    double kineticEnergy() const {
        double ke = 0;
        for (auto* b : {bodyA.get(), bodyB.get()}) {
            auto qd = b->getFlexQd(); auto md = b->getMassDiagonal();
            for (int i = 0; i < (int)qd.size(); i++) ke += 0.5*md[i]*qd[i]*qd[i];
        }
        return ke;
    }
};

// ─── Qt Widget ───────────────────────────────────────────────────────────────
class BarCollisionWidget : public QWidget {
public:
    BarCollisionWidget(QWidget* parent = nullptr) : QWidget(parent) {
        setWindowTitle("MBC++ — Çubuk Çarpışması (Tet Mesh)");
        resize(1200, 750);
        sim_.build();
        timer_ = new QTimer(this);
        connect(timer_, &QTimer::timeout, this, &BarCollisionWidget::tick);
        timer_->start(16);
        elapsed_.start();
        setFocusPolicy(Qt::StrongFocus);
        azimuth_ = 0.6; elevation_ = 0.35;
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this); p.setRenderHint(QPainter::Antialiasing);
        int w = width(), h = height();
        p.fillRect(rect(), QColor(12, 14, 20));

        double scale = zoom_ * std::min(w, h) * 0.6;
        double cx = w*0.5 + panX_, cy = h*0.6 + panY_;
        double ca = std::cos(azimuth_), sa = std::sin(azimuth_);
        double cEl = std::cos(elevation_), sEl = std::sin(elevation_);
        auto toScreen = [&](Vec3 v) -> QPointF {
            double x1 = v.x*ca + v.z*sa, y1 = v.y, z1 = -v.x*sa + v.z*ca;
            return {cx + x1*scale, cy - (y1*cEl - z1*sEl)*scale};
        };

        // Ground grid
        p.setPen(QPen(QColor(28, 32, 44), 1));
        for (double m = -0.5; m <= 0.5 + 1e-6; m += 0.05)
            p.drawLine(toScreen({m, 0, -0.3}), toScreen({m, 0, 0.3}));
        for (double m = -0.3; m <= 0.3 + 1e-6; m += 0.05)
            p.drawLine(toScreen({-0.5, 0, m}), toScreen({0.5, 0, m}));

        // Axes
        auto drawAxis = [&](Vec3 tip, QColor c, const QString& l) {
            p.setPen(QPen(c, 2));
            p.drawLine(toScreen({0,0,0}), toScreen(tip));
            p.setFont(QFont("Sans", 9, QFont::Bold));
            p.drawText(toScreen(tip) + QPointF(4, -4), l);
        };
        drawAxis({0.10,0,0}, QColor(220,70,70), "X");
        drawAxis({0,0.10,0}, QColor(70,200,70), "Y");
        drawAxis({0,0,0.10}, QColor(70,130,255), "Z");

        // Draw body
        auto drawBody = [&](const std::vector<Vec3>& pos,
                            const std::vector<std::array<int,4>>& tets,
                            QColor tint)
        {
            // Collect unique edges
            std::set<std::pair<int,int>> edgeSet;
            for (auto& t : tets) {
                int n[4] = {t[0], t[1], t[2], t[3]};
                for (int a = 0; a < 4; a++)
                    for (int b = a+1; b < 4; b++)
                        edgeSet.insert({std::min(n[a],n[b]), std::max(n[a],n[b])});
            }

            // Faces (sorted by depth)
            static const int fi[4][3] = {{1,2,3},{0,2,3},{0,1,3},{0,1,2}};
            struct Face { QPointF pts[3]; double depth; };
            std::vector<Face> faces;
            for (auto& t : tets) {
                QPointF sp[4];
                for (int k = 0; k < 4; k++) sp[k] = toScreen(pos[t[k]]);
                for (int f = 0; f < 4; f++) {
                    Face fc;
                    fc.depth = 0;
                    for (int k = 0; k < 3; k++) {
                        fc.pts[k] = sp[fi[f][k]];
                        auto& v = pos[t[fi[f][k]]];
                        fc.depth += (-v.x*sa + v.z*ca);
                    }
                    fc.depth /= 3;
                    faces.push_back(fc);
                }
            }
            std::sort(faces.begin(), faces.end(),
                      [](const Face& a, const Face& b) { return a.depth < b.depth; });

            QColor fc(tint.red(), tint.green(), tint.blue(), 140);
            for (auto& f : faces) {
                QPainterPath path;
                path.moveTo(f.pts[0]); path.lineTo(f.pts[1]);
                path.lineTo(f.pts[2]); path.closeSubpath();
                p.fillPath(path, fc);
            }

            // Edges (white wireframe)
            p.setPen(QPen(QColor(255, 255, 255, 200), 1.5));
            for (auto& [a, b] : edgeSet)
                p.drawLine(toScreen(pos[a]), toScreen(pos[b]));
        };

        // Depth-sort bodies
        Vec3 cA = bodyCentroid(sim_.posA), cB = bodyCentroid(sim_.posB);
        if ((-cA.x*sa + cA.z*ca) < (-cB.x*sa + cB.z*ca)) {
            drawBody(sim_.posA, sim_.tetA, QColor(220, 120, 30));   // turuncu
            drawBody(sim_.posB, sim_.tetB, QColor(50, 140, 240));   // mavi
        } else {
            drawBody(sim_.posB, sim_.tetB, QColor(50, 140, 240));
            drawBody(sim_.posA, sim_.tetA, QColor(220, 120, 30));
        }

        drawHUD(p, w, h);
    }

    void keyPressEvent(QKeyEvent* e) override {
        switch (e->key()) {
        case Qt::Key_Space: paused_ = !paused_; break;
        case Qt::Key_R: sim_.build(); elapsed_.restart(); frameCount_ = 0; break;
        case Qt::Key_G: sim_.gravityOn = !sim_.gravityOn; break;
        case Qt::Key_E: {
            auto qd = sim_.bodyB->getFlexQd();
            int nn = sim_.bodyB->nodes.size();
            for (int i = 0; i < nn; i++) { qd[i*12] += 2.0; qd[i*12+2] += 1.0; }
            sim_.bodyB->setFlexQd(qd); break;
        }
        case Qt::Key_Plus: case Qt::Key_Equal: zoom_ *= 1.2; break;
        case Qt::Key_Minus: zoom_ /= 1.2; break;
        case Qt::Key_Escape: close(); break;
        }
        update();
    }
    void mousePressEvent(QMouseEvent* e) override { lastMousePos_ = e->pos(); e->accept(); }
    void mouseMoveEvent(QMouseEvent* e) override {
        QPoint d = e->pos() - lastMousePos_; lastMousePos_ = e->pos();
        if (e->buttons() & Qt::LeftButton) {
            azimuth_ -= d.x()*0.005; elevation_ -= d.y()*0.005;
            elevation_ = std::clamp(elevation_, -1.5, 1.5);
        }
        if (e->buttons() & (Qt::RightButton|Qt::MiddleButton)) { panX_ += d.x(); panY_ += d.y(); }
        update(); e->accept();
    }
    void wheelEvent(QWheelEvent* e) override {
        zoom_ *= (e->angleDelta().y() > 0) ? 1.15 : 1.0/1.15; update(); e->accept();
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
    QTimer* timer_ = nullptr; QElapsedTimer elapsed_;
    bool paused_ = false; int frameCount_ = 0;
    double zoom_ = 1.0, azimuth_ = 0, elevation_ = 0, panX_ = 0, panY_ = 0;
    QPoint lastMousePos_;

    void drawHUD(QPainter& p, int w, int) {
        double fps = frameCount_ / (elapsed_.elapsed()*0.001 + 1e-9);
        int nc = sim_.numContacts();
        p.setPen(QColor(200, 200, 200));
        QFont f("Monospace", 11); f.setStyleHint(QFont::Monospace); p.setFont(f);
        int x = 14, y = 24, dy = 20;
        auto line = [&](const QString& s) { p.drawText(x, y, s); y += dy; };

        line(QString("t = %1 s").arg(sim_.time, 0, 'f', 3));
        line(QString("FPS  %1").arg(fps, 0, 'f', 1));
        line(QString("Substeps/frame: %1").arg(NSUB));
        line(QString("Threads: %1").arg(ThreadConfig::numThreads()));
        line("");

        if (nc > 0) {
            p.setPen(QColor(255, 80, 80));
            QFont fb("Monospace", 12, QFont::Bold); p.setFont(fb);
            p.drawText(x, y, QString("● CONTACT  (%1 pts)").arg(nc));
            y += dy + 2; p.setFont(f); p.setPen(QColor(200, 200, 200));
        } else {
            p.setPen(QColor(80, 200, 80)); line("○ no contact"); p.setPen(QColor(200, 200, 200));
        }
        line("");
        line(QString("A (yatay): %1×%2×%3  E=%4").arg(BAR_A_LX).arg(BAR_A_LY).arg(BAR_A_LZ).arg(EA, 0, 'e', 1));
        line(QString("B (dikey): %1×%2×%3  E=%4").arg(BAR_B_LX).arg(BAR_B_LY).arg(BAR_B_LZ).arg(EB, 0, 'e', 1));
        line(QString("Nodes A: %1  B: %2").arg(sim_.bodyA->nodes.size()).arg(sim_.bodyB->nodes.size()));
        line(QString("DOF  A: %1  B: %2").arg(sim_.bodyA->numDof).arg(sim_.bodyB->numDof));
        line(QString("Tets A: %1  B: %2").arg(sim_.tetA.size()).arg(sim_.tetB.size()));
        line(QString("KE: %1 J").arg(sim_.kineticEnergy(), 0, 'e', 3));
        line(QString("Gravity: %1").arg(sim_.gravityOn ? "ON" : "OFF"));

        // Controls
        p.setPen(QColor(120, 120, 140));
        int bx = w - 260, by = 24;
        auto r = [&](const QString& s) { p.drawText(bx, by, s); by += 18; };
        r("Sol sürükle    Döndür");
        r("Sağ sürükle    Kaydır");
        r("Tekerlek       Zoom");
        r("E              İtki");
        r("G              Gravity");
        r("SPACE          Duraklat");
        r("R              Sıfırla");
        r("ESC            Çıkış");
        p.setPen(QColor(255, 150, 80));
        p.drawText(bx, by + 10, "■ A: yatay çubuk (turuncu)");
        p.setPen(QColor(80, 150, 255));
        p.drawText(bx, by + 28, "■ B: dikey çubuk (mavi, düşen)");

        if (paused_) {
            p.setPen(QColor(255, 80, 80));
            QFont fb("Sans", 16, QFont::Bold); p.setFont(fb);
            p.drawText(w/2 - 50, 40, "⏸  DURAKLATILDI");
        }
    }
};

int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-c" && i+1 < argc)
            ThreadConfig::setNumThreads(std::atoi(argv[++i]));
    }
    printf("[MBC++] Using %d OpenMP thread(s)\n", ThreadConfig::numThreads());
    QApplication app(argc, argv);
    BarCollisionWidget win; win.show();
    return app.exec();
}
