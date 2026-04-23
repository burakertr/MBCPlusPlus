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
#include <QProcess>
#include <QImage>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <cmath>
#include <deque>
#include <vector>
#include <fstream>

#include "mb/core/RigidBody.h"
#include "mb/constraints/SphericalJoint.h"
#include "mb/solvers/NewtonRaphson.h"
#include "mb/integrators/RungeKutta.h"
#include "mb/system/MultibodySystem.h"
#include "mb/core/ThreadConfig.h"
#include "mb/integrators/BDF.h"

#include <cstdlib>

using namespace mb;
static constexpr double DT          = 0.005;   // adım [s]
static constexpr double T_END       = 30.0;    // simülasyon süresi [s]
static constexpr double HHT_ALPHA   =  -0.02;   // daha düşük sayısal sönüm
static constexpr double HHT_RELTOL  = 1e-4;   // Newton göreli tolerans
static constexpr double HHT_ABSTOL  = 1e-6;   // Newton mutlak tolerans
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
    double length2 = 1.0;

    void build(double omega0 = 0) {
        sys = MultibodySystem("DoublePendulum");
        sys.setGravity(Vec3(0, -9.81, 0));

        ground = RigidBody::createGround("Ground");
        sys.addBody(ground);

        // ── Link 1: horizontal, J1 at origin ──────────────────────────────
        // Local Y axis → world +X (rod points right). q = R_z(-90°)
        link1 = RigidBody::createRod(1.0, length1, 0.02, "Link1");
        link1->position    = Vec3(length1 * 0.5, 0, 0);
        link1->orientation = Quaternion(0.7071067811865476, 0, 0, -0.7071067811865476);
        // Give link1 an initial clockwise angular velocity omega0.
        // For J1 (at origin) to stay fixed: v1_y = -L1/2 * omega0
        link1->angularVelocity = Vec3(0, 0, -0);
        link1->velocity        = Vec3(0, -length1 * 0.5 * 0, 0);
        sys.addBody(link1);

        // ── Link 2: also HORIZONTAL, but with ZERO angular velocity ──────
        // Link2 starts in line with link1 (both horizontal, same orientation).
        // Velocity at J2 from link1:
        //   v_J2 = v_CM1 + ω1 × r_J2  where r_J2 = (+L1/2, 0, 0)
        //        = (0, -0.5*ω0, 0) + (0,0,-ω0)×(0.5,0,0)
        //        = (0, -0.5*ω0, 0) + (0, -0.5*ω0, 0) = (0, -ω0, 0)
        // J2 velocity constraint: v_CM2 + ω2×r_link2_J2 = v_J2
        //   with ω2 = 0 and r_link2_J2 = (-L2/2, 0, 0):
        //   → v_CM2 = (0, -ω0, 0)
        //
        // Link1 has angular velocity ω0, link2 has NONE → they diverge
        // immediately from the first integration step.
        link2 = RigidBody::createRod(1.0, length2, 0.02, "Link2");
        link2->position        = Vec3(length1 + length2 * 0.5, 0, 0);
        link2->orientation     = Quaternion(0.7071067811865476, 0, 0, -0.7071067811865476);
        link2->angularVelocity = Vec3::zero();
        link2->velocity        = Vec3(0, -0, 0);
        sys.addBody(link2);

        // ── Constraints ────────────────────────────────────────────────────
        joint1 = std::make_shared<SphericalJoint>(
            ground.get(), link1.get(),
            Vec3(0, 0, 0), Vec3(0, -length1 * 0.5, 0), "J1");
        sys.addConstraint(joint1);

        joint2 = std::make_shared<SphericalJoint>(
            link1.get(), link2.get(),
            Vec3(0, length1 * 0.5, 0), Vec3(0, -length2 * 0.5, 0), "J2");
        sys.addConstraint(joint2);

        SolverConfig solverCfg;
        solverCfg.maxIterations = 25;
        solverCfg.tolerance = 1e-10;
        solverCfg.warmStart = true;
        sys.setSolver(std::make_shared<NewtonRaphsonSolver>(solverCfg));

        // Tight adaptive integrator — important for chaotic double pendulum
        mb::IntegratorConfig cfg;
        cfg.adaptive = false;
        cfg.relTol   = HHT_RELTOL;
        cfg.absTol   = HHT_ABSTOL;
        
        sys.setIntegrator(std::make_shared<HHTAlpha>(HHT_ALPHA, 10, HHT_ABSTOL, cfg));

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
//  Angle Plot Widget (separate window)
// ─────────────────────────────────────────────
class PlotCanvas : public QWidget {
public:
    explicit PlotCanvas(QWidget* parent = nullptr) : QWidget(parent) {
        setMinimumSize(700, 400);
    }

    void setData(const std::vector<double>& t,
                 const std::vector<double>& th1,
                 const std::vector<double>& th2)
    {
        t_ = t; th1_ = th1; th2_ = th2;
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        int W = width(), H = height();
        p.fillRect(rect(), QColor(20, 22, 28));

        if (t_.size() < 2) {
            p.setPen(QColor(180, 180, 180));
            p.drawText(rect(), Qt::AlignCenter, "Simülasyon başlatılıyor...");
            return;
        }

        const int ml = 60, mr = 20, mt = 30, mb = 40;
        int pw = W - ml - mr, ph = H - mt - mb;

        double tMin = t_.front(), tMax = t_.back();
        if (tMax - tMin > 10.0) tMin = tMax - 10.0;

        double yMin = -200.0, yMax = 200.0;

        auto sx = [&](double tt) -> double {
            return ml + (tt - tMin) / (tMax - tMin + 1e-12) * pw;
        };
        auto sy = [&](double deg) -> double {
            return mt + ph - (deg - yMin) / (yMax - yMin) * ph;
        };

        // Grid lines
        p.setPen(QPen(QColor(45, 48, 58), 1));
        for (double deg = -180; deg <= 180; deg += 45) {
            double y = sy(deg);
            p.drawLine(QPointF(ml, y), QPointF(ml + pw, y));
            p.setPen(QColor(100, 100, 120));
            p.drawText(QRectF(0, y - 10, ml - 4, 20), Qt::AlignRight | Qt::AlignVCenter,
                       QString::number((int)deg) + "°");
            p.setPen(QPen(QColor(45, 48, 58), 1));
        }
        p.setPen(QPen(QColor(80, 80, 90), 1, Qt::DashLine));
        p.drawLine(QPointF(ml, sy(0)), QPointF(ml + pw, sy(0)));

        // Time axis ticks
        QFont f("Monospace", 8);
        p.setFont(f);
        for (double tt = std::ceil(tMin); tt <= tMax + 0.01; tt += 1.0) {
            double x = sx(tt);
            if (x < ml || x > ml + pw) continue;
            p.setPen(QPen(QColor(45, 48, 58), 1));
            p.drawLine(QPointF(x, mt), QPointF(x, mt + ph));
            p.setPen(QColor(100, 100, 120));
            p.drawText(QRectF(x - 20, mt + ph + 4, 40, 18), Qt::AlignCenter,
                       QString::number((int)tt) + "s");
        }

        p.setPen(QPen(QColor(100, 100, 120), 1));
        p.drawRect(ml, mt, pw, ph);

        // theta1 (yellow)
        {
            QPainterPath path;
            bool first = true;
            for (size_t i = 0; i < t_.size(); i++) {
                if (t_[i] < tMin) continue;
                double x = sx(t_[i]);
                double y = sy(th1_[i] * 180.0 / M_PI);
                y = std::max((double)mt, std::min((double)(mt + ph), y));
                if (first) { path.moveTo(x, y); first = false; }
                else path.lineTo(x, y);
            }
            p.setPen(QPen(QColor(255, 210, 60), 2));
            p.drawPath(path);
        }

        // theta2 (cyan)
        {
            QPainterPath path;
            bool first = true;
            for (size_t i = 0; i < t_.size(); i++) {
                if (t_[i] < tMin) continue;
                double x = sx(t_[i]);
                double y = sy(th2_[i] * 180.0 / M_PI);
                y = std::max((double)mt, std::min((double)(mt + ph), y));
                if (first) { path.moveTo(x, y); first = false; }
                else path.lineTo(x, y);
            }
            p.setPen(QPen(QColor(60, 200, 255), 2));
            p.drawPath(path);
        }

        // Legend
        QFont lf("Sans", 10, QFont::Bold);
        p.setFont(lf);
        p.fillRect(ml + 10, mt + 8,  20, 4, QColor(255, 210, 60));
        p.setPen(QColor(255, 210, 60));
        p.drawText(ml + 34, mt + 16, "θ₁  (Link 1, dikey'den)");
        p.fillRect(ml + 10, mt + 22, 20, 4, QColor(60, 200, 255));
        p.setPen(QColor(60, 200, 255));
        p.drawText(ml + 34, mt + 30, "θ₂  (Link 2, dikey'den)");

        QFont tf("Sans", 11, QFont::Bold);
        p.setFont(tf);
        p.setPen(QColor(200, 200, 210));
        p.drawText(QRectF(ml, 4, pw, 22), Qt::AlignCenter, "Sarkaç Açıları — Zamana Göre");
    }

private:
    std::vector<double> t_, th1_, th2_;
};

class PlotWindow : public QWidget {
    Q_OBJECT
public:
    explicit PlotWindow(QWidget* parent = nullptr) : QWidget(parent) {
        setWindowTitle("MBC++ — Açı Grafiği");
        resize(800, 480);

        canvas_      = new PlotCanvas(this);
        btnSavePng_  = new QPushButton("PNG Kaydet", this);
        btnSaveCsv_  = new QPushButton("CSV Kaydet", this);
        btnSavePng_->setFixedHeight(30);
        btnSaveCsv_->setFixedHeight(30);

        connect(btnSavePng_, &QPushButton::clicked, this, &PlotWindow::savePng);
        connect(btnSaveCsv_, &QPushButton::clicked, this, &PlotWindow::saveCsv);

        QHBoxLayout* btnLayout = new QHBoxLayout();
        btnLayout->addWidget(btnSavePng_);
        btnLayout->addWidget(btnSaveCsv_);
        btnLayout->addStretch();

        QVBoxLayout* layout = new QVBoxLayout(this);
        layout->addWidget(canvas_);
        layout->addLayout(btnLayout);
    }

    void updatePlot(const std::vector<double>& t,
                    const std::vector<double>& th1,
                    const std::vector<double>& th2)
    {
        t_ = t; th1_ = th1; th2_ = th2;
        canvas_->setData(t, th1, th2);
    }

private slots:
    void savePng() {
        QString path = QFileDialog::getSaveFileName(
            this, "PNG olarak kaydet", "angles.png", "PNG (*.png)");
        if (path.isEmpty()) return;
        QImage img(canvas_->size(), QImage::Format_ARGB32);
        img.fill(Qt::black);
        canvas_->render(&img);
        if (img.save(path))
            QMessageBox::information(this, "Kaydedildi", path + " kaydedildi.");
        else
            QMessageBox::warning(this, "Hata", "Dosya kaydedilemedi.");
    }

    void saveCsv() {
        QString path = QFileDialog::getSaveFileName(
            this, "CSV olarak kaydet", "angles.csv", "CSV (*.csv)");
        if (path.isEmpty()) return;
        std::ofstream ofs(path.toStdString());
        if (!ofs) { QMessageBox::warning(this, "Hata", "Dosya açılamadı."); return; }
        ofs << "time_s,theta1_deg,theta2_deg\n";
        for (size_t i = 0; i < t_.size(); i++)
            ofs << t_[i] << "," << th1_[i]*180.0/M_PI << "," << th2_[i]*180.0/M_PI << "\n";
        QMessageBox::information(this, "Kaydedildi", path + " kaydedildi.");
    }

private:
    PlotCanvas*  canvas_;
    QPushButton* btnSavePng_;
    QPushButton* btnSaveCsv_;
    std::vector<double> t_, th1_, th2_;
};

// ─────────────────────────────────────────────
//  Qt Widget
// ─────────────────────────────────────────────
class PendulumWidget : public QWidget {
    Q_OBJECT
public:
    PendulumWidget(bool record = false, QWidget* parent = nullptr)
        : QWidget(parent), recording_(record)
    {
        setWindowTitle("MBC++ — Çift Sarkaç");
        resize(900, 750);
        setMinimumSize(600, 500);

        sim_.build(3.0);

        // Trace ring-buffer
        traceMax_ = 3000;

        // Plot window
        plotWin_ = new PlotWindow();
        plotWin_->show();

        if (recording_) {
            ffmpeg_ = new QProcess(this);
            QStringList args;
            args << "-y" << "-f" << "rawvideo" << "-pixel_format" << "bgra"
                 << "-video_size" << QString("%1x%2").arg(width()).arg(height())
                 << "-framerate" << "60"
                 << "-i" << "pipe:0"
                 << "-c:v" << "libx264" << "-preset" << "fast"
                 << "-crf" << "18" << "-pix_fmt" << "yuv420p"
                 << "double_pendulum_sim.mp4";
            ffmpeg_->start("ffmpeg", args);
            ffmpeg_->waitForStarted();
            printf("[REC] Recording to double_pendulum_sim.mp4 (%dx%d @60fps)\n", width(), height());
        }

        // Timer → ~60 FPS
        timer_ = new QTimer(this);
        connect(timer_, &QTimer::timeout, this, &PendulumWidget::tick);
        timer_->start(16);

        elapsed_.start();
        setFocusPolicy(Qt::StrongFocus);
    }

    ~PendulumWidget() override {
        stopRecording();
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
            timeData_.clear(); theta1Data_.clear(); theta2Data_.clear();
            sim_.build(2.0 + (std::rand() % 30) * 0.1);
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

            // Compute angles from vertical (downward = negative Y)
            Vec3 pivot = sim_.pivot;
            Vec3 j2World = sim_.link1->bodyToWorld(Vec3(0, sim_.length1 * 0.5, 0));
            Vec3 tip = sim_.tip;

            // theta1: angle of link1 from downward vertical
            double dx1 = j2World.x - pivot.x;
            double dy1 = j2World.y - pivot.y;
            double theta1 = std::atan2(dx1, -dy1);

            // theta2: angle of link2 from downward vertical
            double dx2 = tip.x - j2World.x;
            double dy2 = tip.y - j2World.y;
            double theta2 = std::atan2(dx2, -dy2);

            double t = sim_.sys.getTime();
            timeData_.push_back(t);
            theta1Data_.push_back(theta1);
            theta2Data_.push_back(theta2);

            // Keep at most last 20000 points
            if (timeData_.size() > 20000) {
                timeData_.erase(timeData_.begin(), timeData_.begin() + 1000);
                theta1Data_.erase(theta1Data_.begin(), theta1Data_.begin() + 1000);
                theta2Data_.erase(theta2Data_.begin(), theta2Data_.begin() + 1000);
            }

            // Update plot every ~30 frames
            if (frameCount_ % 30 == 0 && plotWin_)
                plotWin_->updatePlot(timeData_, theta1Data_, theta2Data_);
        }
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
    bool paused_   = false;
    bool dragging_  = false;
    int  frameCount_ = 0;
    int  traceMax_;
    std::deque<Vec3> trace_;
    bool recording_ = false;
    QProcess* ffmpeg_ = nullptr;
    PlotWindow* plotWin_ = nullptr;

    std::vector<double> timeData_, theta1Data_, theta2Data_;

    void stopRecording() {
        if (ffmpeg_ && ffmpeg_->state() == QProcess::Running) {
            ffmpeg_->closeWriteChannel();
            ffmpeg_->waitForFinished(5000);
            printf("[REC] Saved double_pendulum_sim.mp4\n");
            ffmpeg_ = nullptr;
        }
    }

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
        line(QString("E    %1").arg(a.totalEnergy, 16, 'f', 16));
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

#include "pendulum_qt.moc"

// ─────────────────────────────────────────────
int main(int argc, char* argv[]) {
    bool record = false;
    // Parse -c N for thread count
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-c" && i+1 < argc) {
            ThreadConfig::setNumThreads(std::atoi(argv[++i]));
        } else if (std::string(argv[i]) == "-r") {
            record = true;
        }
    }

    QApplication app(argc, argv);
    PendulumWidget win(record);
    win.show();
    return app.exec();
}
