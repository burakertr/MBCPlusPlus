/**
 * 20-link pendulum chain with REVOLUTE JOINTS and viscous damping.
 * All links start horizontal, extending in the +X direction.
 * Revolute axes are in the Z direction — motion constrained to X-Y plane.
 *
 * Controls:
 *   Space  – Pause / Resume
 *   R      – Reset
 *   +/-    – Increase / decrease damping
 *   ESC    – Quit
 */
#include <QApplication>
#include <QWidget>
#include <QPainter>
#include <QPainterPath>
#include <QTimer>
#include <QKeyEvent>
#include <QElapsedTimer>
#include <QFont>
#include <QProcess>
#include <QImage>
#include <cmath>
#include <deque>
#include <vector>
#include <array>

#include "mb/core/RigidBody.h"
#include "mb/constraints/RevoluteJoint.h"
#include "mb/solvers/DirectSolver.h"
#include "mb/integrators/RungeKutta.h"
#include "mb/system/MultibodySystem.h"
#include "mb/core/ThreadConfig.h"
#include <cstdlib>

using namespace mb;

// ─────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────
static constexpr int N_LINKS = 20;
static constexpr double LINK_LENGTH = 0.5;   // metres per link
static constexpr double LINK_MASS   = 0.2;   // kg per link
static constexpr double LINK_RADIUS = 0.05;  // visual rod radius

// ─────────────────────────────────────────────
//  Nice rainbow palette
// ─────────────────────────────────────────────
static QColor linkColor(int i, int n) {
    double t = double(i) / std::max(1, n - 1);
    // HSV rainbow: hue 0..270 (red → blue-violet)
    return QColor::fromHsvF(t * 0.75, 0.85, 0.95);
}

// ─────────────────────────────────────────────
//  Simulation
// ─────────────────────────────────────────────
struct Simulation {
    MultibodySystem sys{"Chain20Rev"};

    std::shared_ptr<RigidBody> ground;
    std::vector<std::shared_ptr<RigidBody>> links;
    std::vector<std::shared_ptr<RevoluteJoint>> joints;

    double dt       = 0.0005;
    int    subSteps = 20;     // 20 × 0.5 ms = 10 ms / frame
    double damping  = 0.05;   // Nm·s/rad — joint damping coefficient

    // Cached world positions: jointPos[i] = joint between link i-1 and link i
    // jointPos[0] = pivot (ground attachment)
    // tipPos      = free end of last link
    std::vector<Vec3> jointPos;
    Vec3 tipPos;

    void build() {
        sys = MultibodySystem("Chain20Rev");
        sys.setGravity(Vec3(0, -9.81, 0));

        ground = RigidBody::createGround("Ground");
        sys.addBody(ground);

        links.clear();
        joints.clear();
        jointPos.resize(N_LINKS + 1);

        // Quaternion for 90° rotation around Z  (local Y → world X)
        // Rotation -90° around Z:  cos(-45°) = 0.7071, sin(-45°) = -0.7071
        Quaternion horizQ(0.7071067811865476, 0, 0, -0.7071067811865476);

        double xAccum = 0.0; // running x-position of joints

        for (int i = 0; i < N_LINKS; i++) {
            double L = LINK_LENGTH;
            double halfL = L * 0.5;

            // Centre of link i
            double cx = xAccum + halfL;
            auto link = RigidBody::createRod(LINK_MASS, L, LINK_RADIUS,
                                             "Link" + std::to_string(i));
            link->position = Vec3(cx, 0, 0);
            link->orientation = horizQ;
            sys.addBody(link);
            links.push_back(link);

            // Constraint: attach to previous body at xAccum
            RigidBody* prevBody = (i == 0)
                ? ground.get()
                : links[i - 1].get();

            Vec3 prevLocal = (i == 0)
                ? Vec3(0, 0, 0)                         // ground origin
                : Vec3(0, halfL, 0);                     // +Y end of previous link

            Vec3 curLocal = Vec3(0, -halfL, 0);          // -Y end of current link

            // Revolute joint: axis = local Z for both bodies
            auto jt = std::make_shared<RevoluteJoint>(
                prevBody, link.get(),
                prevLocal, curLocal,
                Vec3(0, 0, 1), Vec3(0, 0, 1),   // revolute axis = Z
                damping,                          // viscous damping
                "J" + std::to_string(i));
            sys.addConstraint(jt);
            joints.push_back(jt);

            xAccum += L;
        }

        // Integrator: DOPRI45 adaptive
        sys.setSolver(std::make_shared<DirectSolver>());
        IntegratorConfig cfg;
        cfg.adaptive = true;
        cfg.absTol   = 1e-6;
        cfg.relTol   = 1e-4;
        cfg.maxStep  = 0.02;
        cfg.minStep  = 1e-3;
        sys.setIntegrator(std::make_shared<DormandPrince45>(cfg));

        sys.initialize();
        updatePositions();
    }

    void setDamping(double c) {
        damping = c;
        for (auto& j : joints)
            j->setDamping(c);
    }

    void step() {
        for (int i = 0; i < subSteps; i++)
            sys.step(dt);
        updatePositions();
    }

    void updatePositions() {
        // Joint 0 = pivot on ground (attachment of link 0)
        jointPos[0] = links[0]->bodyToWorld(Vec3(0, -LINK_LENGTH * 0.5, 0));

        for (int i = 0; i < N_LINKS; i++) {
            // The +Y end of link i = joint between link i and link i+1
            jointPos[i + 1] = links[i]->bodyToWorld(Vec3(0, LINK_LENGTH * 0.5, 0));
        }
        tipPos = jointPos[N_LINKS];
    }
};

// ─────────────────────────────────────────────
//  Qt Widget
// ─────────────────────────────────────────────
class ChainWidget : public QWidget {
public:
    ChainWidget(bool record = false, QWidget* parent = nullptr)
        : QWidget(parent), recording_(record) {
        setWindowTitle("MBC++ — 20 Zincir (Revolute + Damping)");
        resize(1100, 850);
        setMinimumSize(800, 600);

        sim_.build();
        traceMax_ = 4000;

        if (recording_) {
            ffmpeg_ = new QProcess(this);
            QStringList args;
            args << "-y" << "-f" << "rawvideo" << "-pixel_format" << "bgra"
                 << "-video_size" << QString("%1x%2").arg(width()).arg(height())
                 << "-framerate" << "60"
                 << "-i" << "pipe:0"
                 << "-c:v" << "libx264" << "-preset" << "fast"
                 << "-crf" << "18" << "-pix_fmt" << "yuv420p"
                 << "chain_revolute_sim.mp4";
            ffmpeg_->start("ffmpeg", args);
            ffmpeg_->waitForStarted();
            printf("[REC] Recording to chain_revolute_sim.mp4 (%dx%d @60fps)\n", width(), height());
        }

        timer_ = new QTimer(this);
        connect(timer_, &QTimer::timeout, this, &ChainWidget::tick);
        timer_->start(16);  // ~60 FPS

        elapsed_.start();
        setFocusPolicy(Qt::StrongFocus);
    }

    ~ChainWidget() override {
        stopRecording();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);

        int w = width(), h = height();
        p.fillRect(rect(), QColor(15, 17, 22));

        // Scale: total chain length = N_LINKS * LINK_LENGTH = 6 m
        // We want it to fit nicely on screen
        double totalLen = N_LINKS * LINK_LENGTH;
        double pxPerMeter = std::min(w, h) * 0.38 / totalLen;
        double cx = w * 0.5;
        double cy = h * 0.18;

        auto toScreen = [&](const Vec3& v) -> QPointF {
            return {cx + v.x * pxPerMeter, cy - v.y * pxPerMeter};
        };

        // ── Grid ──
        drawGrid(p, cx, cy, pxPerMeter, w, h);

        // ── Trace of tip ──
        if (trace_.size() >= 2) {
            for (size_t i = 1; i < trace_.size(); i++) {
                double alpha = double(i) / trace_.size();
                QColor c(100, 220, 255, int(alpha * 120));
                p.setPen(QPen(c, 1.2));
                p.drawLine(toScreen(trace_[i - 1]), toScreen(trace_[i]));
            }
        }

        // ── Links ──
        for (int i = 0; i < N_LINKS; i++) {
            QPointF a = toScreen(sim_.jointPos[i]);
            QPointF b = toScreen(sim_.jointPos[i + 1]);

            QColor col = linkColor(i, N_LINKS);
            p.setPen(QPen(col, 4.0, Qt::SolidLine, Qt::RoundCap));
            p.drawLine(a, b);
        }

        // ── Joints ──
        for (int i = 0; i <= N_LINKS; i++) {
            QPointF pt = toScreen(sim_.jointPos[i]);
            double r = (i == 0) ? 7 : 4;

            p.setPen(Qt::NoPen);
            if (i == 0) {
                // Pivot
                p.setBrush(QColor(255, 100, 100));
                p.drawEllipse(pt, r, r);
                p.setBrush(QColor(40, 42, 50));
                p.drawEllipse(pt, r * 0.4, r * 0.4);
            } else if (i == N_LINKS) {
                // Tip
                p.setBrush(QColor(100, 220, 255));
                p.drawEllipse(pt, 6, 6);
            } else {
                // Internal revolute joint — draw as small circle with axis indicator
                QColor jc = linkColor(i, N_LINKS);
                jc = jc.darker(130);
                p.setBrush(jc);
                p.drawEllipse(pt, r, r);
                // small dot center to indicate revolute
                p.setBrush(QColor(255, 255, 255, 120));
                p.drawEllipse(pt, 1.5, 1.5);
            }
        }

        // ── Ceiling / mount ──
        QPointF sPivot = toScreen(sim_.jointPos[0]);
        p.setPen(QPen(QColor(160, 160, 160), 2));
        p.drawLine(QPointF(sPivot.x() - 25, sPivot.y()),
                   QPointF(sPivot.x() + 25, sPivot.y()));
        for (int i = -2; i <= 2; i++) {
            double bx = sPivot.x() + i * 10;
            p.drawLine(QPointF(bx, sPivot.y()), QPointF(bx - 5, sPivot.y() - 7));
        }

        // ── HUD ──
        drawHUD(p, w, h);
    }

    void keyPressEvent(QKeyEvent* e) override {
        if (e->key() == Qt::Key_Space) {
            paused_ = !paused_;
        } else if (e->key() == Qt::Key_R) {
            trace_.clear();
            sim_.build();
            elapsed_.restart();
            frameCount_ = 0;
        } else if (e->key() == Qt::Key_Plus || e->key() == Qt::Key_Equal) {
            sim_.setDamping(sim_.damping * 2.0);
        } else if (e->key() == Qt::Key_Minus) {
            sim_.setDamping(std::max(0.0, sim_.damping * 0.5));
        } else if (e->key() == Qt::Key_0) {
            sim_.setDamping(0.0);
        } else if (e->key() == Qt::Key_Escape) {
            stopRecording();
            close();
        }
    }

private slots:
    void tick() {
        if (!paused_) {
            sim_.step();
            trace_.push_back(sim_.tipPos);
            if ((int)trace_.size() > traceMax_)
                trace_.pop_front();
        }
        frameCount_++;
        update();

        // Write frame to ffmpeg pipe
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
    int  frameCount_ = 0;
    int  traceMax_;
    bool recording_ = false;
    QProcess* ffmpeg_ = nullptr;

    void stopRecording() {
        if (ffmpeg_ && ffmpeg_->state() == QProcess::Running) {
            ffmpeg_->closeWriteChannel();
            ffmpeg_->waitForFinished(5000);
            printf("[REC] Saved chain_revolute_sim.mp4\n");
            ffmpeg_ = nullptr;
        }
    }
    std::deque<Vec3> trace_;

    void drawGrid(QPainter& p, double cx, double cy, double ppm, int w, int h) {
        p.setPen(QPen(QColor(35, 38, 48), 1));
        double step = 0.5; // metres
        if (ppm * step < 20) step = 1.0;
        if (ppm * step < 20) step = 2.0;

        for (double m = -20; m <= 20; m += step) {
            double sx = cx + m * ppm;
            if (sx >= 0 && sx <= w)
                p.drawLine(QPointF(sx, 0), QPointF(sx, h));
        }
        for (double m = -20; m <= 20; m += step) {
            double sy = cy - m * ppm;
            if (sy >= 0 && sy <= h)
                p.drawLine(QPointF(0, sy), QPointF(w, sy));
        }
    }

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
        line(QString("%1 link zincir  (Revolute)").arg(N_LINKS));
        line(QString("Damping  %1 Nm·s/rad").arg(sim_.damping, 0, 'f', 4));
        line(QString("KE  %1   PE  %2").arg(a.kineticEnergy, 8, 'f', 4).arg(a.potentialEnergy, 8, 'f', 4));
        line(QString("E   %1").arg(a.totalEnergy, 10, 'f', 6));
        line(QString("|C| %1").arg(a.constraintViolation, 0, 'e', 2));
        line(QString("|Cv| %1").arg(a.velocityViolation, 0, 'e', 2));

        // Controls
        p.setPen(QColor(120, 120, 140));
        QFont f2("Sans", 9);
        p.setFont(f2);
        int bx = w - 250, by = 24;
        auto rline = [&](const QString& s) { p.drawText(bx, by, s); by += 18; };
        rline("SPACE   Duraklat / Devam");
        rline("R       Sıfırla");
        rline("+/-     Damping artır / azalt");
        rline("0       Damping = 0");
        rline("ESC     Çıkış");

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
    bool record = false;
    double damping = 0.05;  // default
    // Parse -c N for thread count, -r for recording, -d for damping
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-c" && i+1 < argc) {
            ThreadConfig::setNumThreads(std::atoi(argv[++i]));
        } else if (std::string(argv[i]) == "-r") {
            record = true;
        } else if (std::string(argv[i]) == "-d" && i+1 < argc) {
            damping = std::atof(argv[++i]);
        }
    }
    printf("[MBC++] Using %d OpenMP thread(s), damping = %.4f Nm·s/rad\n",
           ThreadConfig::numThreads(), damping);

    QApplication app(argc, argv);
    ChainWidget win(record);
    win.show();
    return app.exec();
}
