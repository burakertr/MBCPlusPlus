/**
 * Double-pendulum with real-time Qt visualization.
 * 
 * Features:
 *   - 2D rendering of pendulum links, joints, and tip trace
 *   - Real-time energy / constraint violation overlay
 *   - Pause / resume with Space, reset with R
 *   - Drag link-2 tip with mouse to set initial conditions
 */
#include <QApplication>
#include <QWidget>
#include <QPainter>
#include <QPainterPath>
#include <QTimer>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QElapsedTimer>
#include <QFont>
#include <cmath>
#include <deque>

#include "mb/core/RigidBody.h"
#include "mb/constraints/SphericalJoint.h"
#include "mb/solvers/DirectSolver.h"
#include "mb/integrators/RungeKutta.h"
#include "mb/system/MultibodySystem.h"

using namespace mb;

// ─────────────────────────────────────────────
//  Simulation wrapper
// ─────────────────────────────────────────────
struct Simulation {
    MultibodySystem sys{"DoublePendulum"};

    std::shared_ptr<RigidBody> ground, link1, link2;
    std::shared_ptr<SphericalJoint> joint1, joint2;

    double dt       = 0.0005;   // physics time-step
    int    subSteps = 20;       // sub-steps per frame  (20 × 0.5 ms = 10 ms / frame)

    // Cached world-space positions for rendering
    Vec3 pivot{0, 0, 0};
    Vec3 p1, p2;               // joint positions (link centres)
    Vec3 tip;                  // end of link2

    double length1 = 1.0;
    double length2 = 0.5;

    void build(double omega0 = 3.0) {
        sys = MultibodySystem("DoublePendulum");
        sys.setGravity(Vec3(0, -9.81, 0));

        ground = RigidBody::createGround("Ground");
        sys.addBody(ground);

        link1 = RigidBody::createRod(1.0, length1, 0.02, "Link1");
        link1->position = Vec3(length1 * 0.5, 0, 0);
        // Rotate 90° around Z so rod's local Y axis aligns with world X
        link1->orientation = Quaternion(0.7071067811865476, 0, 0, -0.7071067811865476);
        sys.addBody(link1);

        link2 = RigidBody::createRod(0.5, length2, 0.02, "Link2");
        link2->position = Vec3(length1 + length2 * 0.5, 0, 0);
        link2->orientation = Quaternion(0.7071067811865476, 0, 0, -0.7071067811865476);
        sys.addBody(link2);

        // Joint attachment points in local body frame (Y is rod axis)
        joint1 = std::make_shared<SphericalJoint>(
            ground.get(), link1.get(),
            Vec3(0, 0, 0), Vec3(0, -length1 * 0.5, 0), "J1");
        sys.addConstraint(joint1);

        joint2 = std::make_shared<SphericalJoint>(
            link1.get(), link2.get(),
            Vec3(0, length1 * 0.5, 0), Vec3(0, -length2 * 0.5, 0), "J2");
        sys.addConstraint(joint2);

        // Gravity handled by sys.setGravity() via assembleForces()

        sys.setSolver(std::make_shared<DirectSolver>());
        mb::IntegratorConfig interconfig;
        interconfig.absTol=1e-4;
        interconfig.adaptive=true;
        interconfig.maxStep=0.1;
        interconfig.relTol=1e-6;
        interconfig.minStep=1e-5;
        sys.setIntegrator(std::make_shared<DormandPrince45>(interconfig));

        link1->angularVelocity = Vec3(0, 0, 0);

        sys.initialize();
        updatePositions();
    }

    void step() {
        for (int i = 0; i < subSteps; i++)
            sys.step(dt);
        updatePositions();
    }

    void updatePositions() {
        p1  = link1->position;
        p2  = link2->position;
        // pivot = J1 attachment on link1 = local (0, -L1/2, 0)
        pivot = link1->bodyToWorld(Vec3(0, -length1 * 0.5, 0));
        // j2 = J2 attachment on link1 = local (0, +L1/2, 0)  (computed in paintEvent)
        // tip = free end of link2 = local (0, +L2/2, 0)
        tip = link2->bodyToWorld(Vec3(0, length2 * 0.5, 0));
    }
};

// ─────────────────────────────────────────────
//  Qt Widget
// ─────────────────────────────────────────────
class PendulumWidget : public QWidget {
public:
    PendulumWidget(QWidget* parent = nullptr) : QWidget(parent) {
        setWindowTitle("MBC++ — Çift Sarkaç");
        resize(900, 750);
        setMinimumSize(600, 500);

        sim_.build(3.0);

        // Trace ring-buffer
        traceMax_ = 3000;

        // Timer → ~60 FPS
        timer_ = new QTimer(this);
        connect(timer_, &QTimer::timeout, this, &PendulumWidget::tick);
        timer_->start(16);

        elapsed_.start();
        setFocusPolicy(Qt::StrongFocus);
    }

protected:
    // ── Paint ──
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);

        int w = width(), h = height();

        // Dark background
        p.fillRect(rect(), QColor(20, 22, 28));

        // World → screen transform: 1 m ≈ pxPerMeter pixels
        double pxPerMeter = std::min(w, h) * 0.22;
        double cx = w * 0.5;
        double cy = h * 0.35;

        auto toScreen = [&](const Vec3& v) -> QPointF {
            return {cx + v.x * pxPerMeter, cy - v.y * pxPerMeter};
        };

        // ── Grid ──
        drawGrid(p, cx, cy, pxPerMeter, w, h);

        // ── Trace ──
        if (trace_.size() >= 2) {
            for (size_t i = 1; i < trace_.size(); i++) {
                double alpha = double(i) / trace_.size();
                QColor c(100, 220, 255, int(alpha * 180));
                p.setPen(QPen(c, 1.5));
                p.drawLine(toScreen(trace_[i - 1]), toScreen(trace_[i]));
            }
        }

        // Positions
        QPointF sPivot = toScreen(sim_.pivot);
        QPointF sP1    = toScreen(sim_.p1);
        QPointF sP2    = toScreen(sim_.p2);
        QPointF sTip   = toScreen(sim_.tip);

        // J2 = positive-Y end of link1 (where link2 attaches)
        Vec3 j2World = sim_.link1->bodyToWorld(Vec3(0, sim_.length1 * 0.5, 0));
        QPointF sJ2  = toScreen(j2World);

        // ── Links ──
        // Link 1: pivot → j2
        {
            QPen pen(QColor(220, 180, 60), 6, Qt::SolidLine, Qt::RoundCap);
            p.setPen(pen);
            p.drawLine(sPivot, sJ2);
        }
        // Link 2: j2 → tip
        {
            QPen pen(QColor(60, 180, 220), 6, Qt::SolidLine, Qt::RoundCap);
            p.setPen(pen);
            p.drawLine(sJ2, sTip);
        }

        // ── Joints ──
        auto drawJoint = [&](QPointF pt, double r, QColor fill) {
            p.setPen(Qt::NoPen);
            p.setBrush(fill);
            p.drawEllipse(pt, r, r);
            p.setBrush(QColor(40, 42, 50));
            p.drawEllipse(pt, r * 0.4, r * 0.4);
        };

        drawJoint(sPivot, 10, QColor(255, 100, 100));
        drawJoint(sJ2,    8,  QColor(255, 200, 80));

        // ── Tip mass ──
        p.setPen(Qt::NoPen);
        p.setBrush(QColor(100, 220, 255));
        p.drawEllipse(sTip, 7, 7);

        // ── Ceiling / mount ──
        p.setPen(QPen(QColor(160, 160, 160), 2));
        p.drawLine(QPointF(sPivot.x() - 30, sPivot.y()), QPointF(sPivot.x() + 30, sPivot.y()));
        for (int i = -3; i <= 3; i++) {
            double bx = sPivot.x() + i * 10;
            p.drawLine(QPointF(bx, sPivot.y()), QPointF(bx - 6, sPivot.y() - 8));
        }

        // ── HUD ──
        drawHUD(p, w, h);
    }

    // ── Keyboard ──
    void keyPressEvent(QKeyEvent* e) override {
        if (e->key() == Qt::Key_Space) {
            paused_ = !paused_;
        } else if (e->key() == Qt::Key_R) {
            trace_.clear();
            sim_.build(3.0 + (std::rand() % 40) * 0.1);
            elapsed_.restart();
            frameCount_ = 0;
        } else if (e->key() == Qt::Key_Escape) {
            close();
        }
    }

    // ── Mouse: drag tip ──
    void mousePressEvent(QMouseEvent* e) override {
        if (e->button() == Qt::LeftButton) {
            dragging_ = true;
            paused_ = true;
        }
    }
    void mouseReleaseEvent(QMouseEvent* e) override {
        if (e->button() == Qt::LeftButton) {
            dragging_ = false;
            paused_ = false;
            trace_.clear();
        }
    }
    void mouseMoveEvent(QMouseEvent* e) override {
        if (!dragging_) return;
        // Screen → world
        int w = width(), h = height();
        double pxPerMeter = std::min(w, h) * 0.22;
        double cx = w * 0.5, cy = h * 0.35;
        double wx = (e->pos().x() - cx) / pxPerMeter;
        double wy = -(e->pos().y() - cy) / pxPerMeter;

        // Simple IK: place link2 tip at mouse, try to solve 2-link IK
        double L1 = sim_.length1, L2 = sim_.length2;
        double dist = std::sqrt(wx * wx + wy * wy);
        if (dist > L1 + L2 - 0.01) dist = L1 + L2 - 0.01;
        if (dist < std::abs(L1 - L2) + 0.01) dist = std::abs(L1 - L2) + 0.01;

        // Law of cosines for elbow angle
        double cosA2 = (L1 * L1 + L2 * L2 - dist * dist) / (2 * L1 * L2);
        cosA2 = std::max(-1.0, std::min(1.0, cosA2));
        double a2 = std::acos(cosA2);

        double cosA1 = (L1 * L1 + dist * dist - L2 * L2) / (2 * L1 * dist);
        cosA1 = std::max(-1.0, std::min(1.0, cosA1));
        double a1 = std::atan2(wy, wx) + std::acos(cosA1);

        // Elbow
        double ex = L1 * std::cos(a1);
        double ey = L1 * std::sin(a1);

        // Set body positions / orientations
        sim_.link1->position = Vec3(ex * 0.5, ey * 0.5, 0);
        double ang1 = a1 - M_PI / 2.0;
        sim_.link1->orientation = Quaternion::fromAxisAngle(Vec3(0, 0, 1), ang1);
        sim_.link1->velocity = Vec3::zero();
        sim_.link1->angularVelocity = Vec3::zero();

        double ang2 = std::atan2(wy - ey, wx - ex) - M_PI / 2.0;
        sim_.link2->position = Vec3((ex + wx) * 0.5, (ey + wy) * 0.5, 0);
        sim_.link2->orientation = Quaternion::fromAxisAngle(Vec3(0, 0, 1), ang2);
        sim_.link2->velocity = Vec3::zero();
        sim_.link2->angularVelocity = Vec3::zero();

        sim_.sys.initialize();
        sim_.updatePositions();
        update();
    }

private slots:
    void tick() {
        if (!paused_) {
            sim_.step();
            // Record trace
            trace_.push_back(sim_.tip);
            if ((int)trace_.size() > traceMax_)
                trace_.pop_front();
        }
        frameCount_++;
        update();
    }

private:
    Simulation sim_;
    QTimer* timer_ = nullptr;
    QElapsedTimer elapsed_;
    bool paused_   = false;
    bool dragging_  = false;
    int  frameCount_ = 0;
    int  traceMax_;
    std::deque<Vec3> trace_;

    // ── Grid ──
    void drawGrid(QPainter& p, double cx, double cy, double ppm, int w, int h) {
        p.setPen(QPen(QColor(45, 48, 58), 1));
        // Vertical lines every 0.5 m
        for (double m = -5; m <= 5; m += 0.5) {
            double sx = cx + m * ppm;
            if (sx >= 0 && sx <= w)
                p.drawLine(QPointF(sx, 0), QPointF(sx, h));
        }
        for (double m = -5; m <= 5; m += 0.5) {
            double sy = cy - m * ppm;
            if (sy >= 0 && sy <= h)
                p.drawLine(QPointF(0, sy), QPointF(w, sy));
        }
    }

    // ── Heads-up display ──
    void drawHUD(QPainter& p, int w, int /*h*/) {
        auto a = sim_.sys.analyze();
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

        line(QString("t = %1 s").arg(sim_.sys.getTime(), 0, 'f', 3));
        line(QString("FPS  %1").arg(fps, 0, 'f', 1));
        line(QString("KE   %1  PE  %2").arg(a.kineticEnergy, 8, 'f', 4).arg(a.potentialEnergy, 8, 'f', 4));
        line(QString("E    %1").arg(a.totalEnergy, 8, 'f', 4));
        line(QString("|C|  %1").arg(a.constraintViolation, 0, 'e', 2));

        // Controls
        p.setPen(QColor(120, 120, 140));
        QFont f2("Sans", 9);
        p.setFont(f2);
        int bx = w - 220, by = 24;
        auto rline = [&](const QString& s) { p.drawText(bx, by, s); by += 18; };
        rline("SPACE  Duraklat / Devam");
        rline("R      Sıfırla");
        rline("Mouse  Sürükle (IK)");
        rline("ESC    Çıkış");

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
    PendulumWidget win;
    win.show();
    return app.exec();
}
