/**
 * vehicle_qt — interactive 4-wheel sedan on a flat road.
 *
 * Controls:
 *   W / S         throttle / brake
 *   A / D         steer left / right
 *   Space         hand-brake (full brake, zero throttle)
 *   R             reset
 *   Esc           quit
 *
 * View: top-down (X = right, Z = down on screen) + side panel HUD with
 * speed, rpm, throttle/brake/steer bars, per-wheel slip and Fz.
 */
#include <QApplication>
#include <QWidget>
#include <QPainter>
#include <QPainterPath>
#include <QPolygonF>
#include <QTimer>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QElapsedTimer>
#include <QFont>
#include <QLinearGradient>
#include <cmath>
#include <deque>
#include <array>
#include <vector>
#include <algorithm>

#include "mb/system/MultibodySystem.h"
#include "mb/solvers/NewtonRaphson.h"
#include "mb/integrators/RungeKutta.h"
#include "mb/integrators/BDF.h"
#include "mb/vehicle/Vehicle.h"
#include "mb/vehicle/Road.h"

using namespace mb;
static constexpr double HHT_ALPHA   =  -0.3;   // daha düşük sayısal sönüm
static constexpr double HHT_RELTOL  = 1e-4;   // Newton göreli tolerans
static constexpr double HHT_ABSTOL  = 1e-6;   // Newton mutlak tolerans

// ──────────────────────────────────────────────────────────────────────────
//  3D orbit camera (software-rendered, QPainter polygons).
//  Convention: world Y is up. Camera looks at `target` from spherical
//  (az, el, dist). az=0 → camera on +Z, looking toward -Z.
// ──────────────────────────────────────────────────────────────────────────
struct Camera3D {
    double az    = 0.7;     // azimuth (yaw around world Y), radians
    double el    = 0.45;    // elevation above horizon, radians
    double dist  = 8.0;     // distance from target (m)
    double fovY  = 55.0 * M_PI / 180.0;
    Vec3   target{0,0,0};
    int    W = 1200, H = 800;

    // Camera basis vectors (right-handed). forward points from camera→target.
    Vec3 forward() const {
        return { -std::sin(az)*std::cos(el),
                 -std::sin(el),
                 -std::cos(az)*std::cos(el) };
    }
    Vec3 right() const {
        return { std::cos(az), 0.0, -std::sin(az) };
    }
    Vec3 up() const {
        // up = right × forward (right-handed)
        Vec3 f = forward(), r = right();
        return { r.y*f.z - r.z*f.y,
                 r.z*f.x - r.x*f.z,
                 r.x*f.y - r.y*f.x };
    }
    Vec3 eye() const {
        Vec3 f = forward();
        return { target.x - f.x*dist, target.y - f.y*dist, target.z - f.z*dist };
    }

    // Returns (sx, sy, depth). depth>0 → in front of camera.
    struct Proj { double sx, sy, depth; bool valid; };
    Proj project(const Vec3& w) const {
        Vec3 e = eye();
        Vec3 d{ w.x - e.x, w.y - e.y, w.z - e.z };
        Vec3 f = forward(), r = right(), u = up();
        double zc = d.x*f.x + d.y*f.y + d.z*f.z;
        double xc = d.x*r.x + d.y*r.y + d.z*r.z;
        double yc = d.x*u.x + d.y*u.y + d.z*u.z;
        if (zc < 1e-3) return {0,0,zc,false};
        double fy = 1.0 / std::tan(fovY * 0.5);
        double aspect = double(W) / double(H);
        double fx = fy / aspect;
        double sx = (xc * fx / zc) * (W * 0.5) + W * 0.5;
        double sy = -(yc * fy / zc) * (H * 0.5) + H * 0.5;
        return {sx, sy, zc, true};
    }
};
// ──────────────────────────────────────────────────────────────────────────
//  Simulation
// ──────────────────────────────────────────────────────────────────────────
struct VehicleSim {
    MultibodySystem sys{"VehicleDemo"};
    std::shared_ptr<RigidBody> ground;
    FlatRoad road{0.0, 1.0};
    std::unique_ptr<Vehicle> vehicle;

    double dt = 0.0001;
    int subSteps = 100;          // 10 ms per frame
    DriverInput input;

    void build() {
        sys = MultibodySystem("VehicleDemo");
        sys.setGravity({0.0, -9.81, 0.0});

        ground = RigidBody::createGround("Ground");
        sys.addBody(ground);

        VehicleParams vp = Vehicle::sedanDefaults("Sedan");
        vp.initialPosition = {0.0, 0.7, 0.0};
        vehicle = std::make_unique<Vehicle>(vp);
        vehicle->attachToSystem(sys, road);

        // Solver / integrator: HHT-α with mild numerical damping for stability
        // of the highly stiff tyre-vertical & spring system. Drive/brake/Mz
        // reactions are now properly applied to the knuckle (Newton 3) and
        // rear corners are PD-locked at 0°, so HHT no longer destabilises.
        SolverConfig solverCfg;
        solverCfg.maxIterations = 20;
        solverCfg.tolerance = 1e-8;
        solverCfg.warmStart = true;
        sys.setSolver(std::make_shared<NewtonRaphsonSolver>(solverCfg));

        IntegratorConfig cfg;
        cfg.adaptive = false;
        cfg.relTol = 1e-4;
        cfg.absTol = 1e-6;
        // BDF2 chosen over HHT-α: with driven wheels at low spin the
        // HHT KKT becomes poorly conditioned (its finite-difference
        // Jacobian generates spurious NaNs as soon as engine torque
        // turns on). BDF2 is unconditionally stable and handles this
        // case cleanly while still providing the implicit step needed
        // by the stiff suspension/contact dynamics.
        sys.setIntegrator(std::make_shared<BDF2>(cfg));

        sys.initialize();
    }

    void step() {
        for (int i = 0; i < subSteps; ++i) {
            vehicle->setManualInput(input);
            vehicle->update(sys.getTime(), dt);
            sys.step(dt);
        }
    }
};

// ──────────────────────────────────────────────────────────────────────────
//  Window
// ──────────────────────────────────────────────────────────────────────────
class VehicleWindow : public QWidget {
public:
    VehicleWindow() {
        setWindowTitle("MBC++ Vehicle Demo");
        resize(1200, 800);
        setFocusPolicy(Qt::StrongFocus);
        sim_.build();
        timer_ = new QTimer(this);
        connect(timer_, &QTimer::timeout, this, [this]{ tick(); });
        timer_->start(10); // 100 Hz wall-clock
        elapsed_.start();
    }

protected:
    void tick() {
        if (!paused_) sim_.step();
        // Push updated trace position.
        Vec3 p = sim_.vehicle->chassis()->position;
        trace_.emplace_back(p.x, p.z);
        if (trace_.size() > 4000) trace_.pop_front();
        update();
    }

    void keyPressEvent(QKeyEvent* e) override {
        switch (e->key()) {
        case Qt::Key_W: sim_.input.throttle = +5.0; break;
        case Qt::Key_S: sim_.input.throttle = -5.0; break;
        case Qt::Key_A: sim_.input.steering = -1.0; break;
        case Qt::Key_D: sim_.input.steering = +1.0; break;
        case Qt::Key_Space: sim_.input.brake = 1.0; sim_.input.throttle = 0.0; break;
        case Qt::Key_R: sim_.build(); trace_.clear(); break;
        case Qt::Key_P: paused_ = !paused_; break;
        // Quick camera presets (relative to chassis +X forward).
        case Qt::Key_1: cam_.az = 0.0;        cam_.el = 1.45;  cam_.dist = 10.0; break; // top
        case Qt::Key_2: cam_.az = M_PI*0.5;   cam_.el = 0.05;  cam_.dist = 8.0;  break; // side
        case Qt::Key_3: cam_.az = M_PI;       cam_.el = 0.25;  cam_.dist = 8.0;  break; // chase
        case Qt::Key_0: cam_.az = 0.7; cam_.el = 0.45; cam_.dist = 8.0; break;
        case Qt::Key_Escape: close(); break;
        }
    }

    void mousePressEvent(QMouseEvent* e) override { lastMouse_ = e->pos(); }
    void mouseMoveEvent(QMouseEvent* e) override {
        QPoint d = e->pos() - lastMouse_;
        lastMouse_ = e->pos();
        if (e->buttons() & Qt::LeftButton) {
            cam_.az -= d.x() * 0.008;
            cam_.el  = std::max(-1.45, std::min(1.45, cam_.el + d.y() * 0.008));
        }
        update();
    }
    void wheelEvent(QWheelEvent* e) override {
        double k = (e->angleDelta().y() > 0) ? 1.0 / 1.15 : 1.15;
        cam_.dist = std::max(1.5, std::min(80.0, cam_.dist * k));
        update();
    }
    void resizeEvent(QResizeEvent*) override {
        cam_.W = width() - 380; cam_.H = height();
    }
    void keyReleaseEvent(QKeyEvent* e) override {
        switch (e->key()) {
        case Qt::Key_W:
        case Qt::Key_S: sim_.input.throttle = 0.0; break;
        case Qt::Key_A:
        case Qt::Key_D: sim_.input.steering = 0.0; break;
        case Qt::Key_Space: sim_.input.brake = 0.0; break;
        }
    }

    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        const int W = width(), H = height();
        p.fillRect(rect(), QColor(30, 32, 38));

        // Layout: left = 3D scene, right = HUD.
        const int hudW = 380;
        const int viewW = W - hudW;
        QRect viewRect(0, 0, viewW, H);
        QRect hudRect(viewW, 0, hudW, H);
        cam_.W = viewW; cam_.H = H;

        drawScene3D(p, viewRect);
        drawHUD(p, hudRect);
    }

    // ----- 3D rendering ---------------------------------------------------
    struct Face {
        std::array<QPointF,4> pts;
        int n;          // 3 or 4
        double depth;   // mean camera-space z (smaller = nearer)
        QColor col;
        bool   outline;
    };

    // Add an oriented box's 6 faces to `faces` (back-face culled, lit).
    void emitBox(std::vector<Face>& faces, const Vec3& center,
                 const Vec3& ax, const Vec3& ay, const Vec3& az,
                 const Vec3& halfExt, const QColor& base,
                 bool outline = true) {
        // 8 corners.
        Vec3 c[8];
        for (int i = 0; i < 8; ++i) {
            double sx = (i & 1) ? halfExt.x : -halfExt.x;
            double sy = (i & 2) ? halfExt.y : -halfExt.y;
            double sz = (i & 4) ? halfExt.z : -halfExt.z;
            c[i] = { center.x + ax.x*sx + ay.x*sy + az.x*sz,
                     center.y + ax.y*sx + ay.y*sy + az.y*sz,
                     center.z + ax.z*sx + ay.z*sy + az.z*sz };
        }
        // 6 faces, vertex indices CCW seen from outside.
        static const int F[6][4] = {
            {0,2,6,4},   // -X
            {1,5,7,3},   // +X
            {0,4,5,1},   // -Y
            {2,3,7,6},   // +Y
            {0,1,3,2},   // -Z
            {4,6,7,5}    // +Z
        };
        // Outward normals in world frame.
        Vec3 normals[6] = {
            {-ax.x,-ax.y,-ax.z}, {ax.x,ax.y,ax.z},
            {-ay.x,-ay.y,-ay.z}, {ay.x,ay.y,ay.z},
            {-az.x,-az.y,-az.z}, {az.x,az.y,az.z}
        };
        Vec3 fwd = cam_.forward();
        const double LX = 0.30, LY = 0.85, LZ = 0.40;
        const double LL = std::sqrt(LX*LX + LY*LY + LZ*LZ);
        for (int fi = 0; fi < 6; ++fi) {
            const Vec3& n = normals[fi];
            // Back-face cull: skip faces whose outward normal points away from camera.
            if (n.x*fwd.x + n.y*fwd.y + n.z*fwd.z > -0.02) continue;
            Face f; f.n = 4; f.outline = outline;
            double dsum = 0; bool ok = true;
            for (int k = 0; k < 4; ++k) {
                auto pr = cam_.project(c[F[fi][k]]);
                if (!pr.valid) { ok = false; break; }
                f.pts[k] = QPointF(pr.sx, pr.sy);
                dsum += pr.depth;
            }
            if (!ok) continue;
            f.depth = dsum * 0.25;
            double lambert = std::max(0.0, -(n.x*LX + n.y*LY + n.z*LZ) / LL);
            double b = 0.30 + 0.70 * lambert;
            f.col = QColor(
                std::min(255, int(base.red()   * b)),
                std::min(255, int(base.green() * b)),
                std::min(255, int(base.blue()  * b)));
            faces.push_back(f);
        }
    }

    void drawScene3D(QPainter& p, const QRect& r) {
        p.save();
        p.setClipRect(r);

        // Sky gradient.
        QLinearGradient sky(r.topLeft(), r.bottomLeft());
        sky.setColorAt(0.0, QColor(45, 60, 90));
        sky.setColorAt(0.55, QColor(110, 135, 160));
        sky.setColorAt(0.551, QColor(70, 90, 60));   // horizon
        sky.setColorAt(1.0, QColor(40, 55, 35));
        p.fillRect(r, sky);

        // Camera follows chassis position.
        Vec3 cpos = sim_.vehicle->chassis()->position;
        cam_.target = cpos;

        // ---- Ground grid (3D lines on y=0). Drawn first so geometry overdraws. ----
        p.setPen(QPen(QColor(60, 80, 60), 1));
        const int gridR = 30;     // m radius around camera target
        const int step  = 1;
        double gx0 = std::floor(cpos.x) - gridR;
        double gz0 = std::floor(cpos.z) - gridR;
        for (int i = 0; i <= 2*gridR; i += step) {
            // X-aligned lines (vary z)
            Vec3 a{gx0,            0.0, gz0 + i};
            Vec3 b{gx0 + 2*gridR,  0.0, gz0 + i};
            auto pa = cam_.project(a), pb = cam_.project(b);
            if (pa.valid && pb.valid)
                p.drawLine(QPointF(pa.sx, pa.sy), QPointF(pb.sx, pb.sy));
            // Z-aligned lines (vary x)
            Vec3 c{gx0 + i, 0.0, gz0};
            Vec3 d{gx0 + i, 0.0, gz0 + 2*gridR};
            auto pc = cam_.project(c), pd = cam_.project(d);
            if (pc.valid && pd.valid)
                p.drawLine(QPointF(pc.sx, pc.sy), QPointF(pd.sx, pd.sy));
        }
        // World axes at origin.
        auto drawAxis = [&](const Vec3& dir, const QColor& col) {
            Vec3 a{0,0.01,0}, b{dir.x, 0.01 + dir.y, dir.z};
            auto pa = cam_.project(a), pb = cam_.project(b);
            if (pa.valid && pb.valid) {
                p.setPen(QPen(col, 2));
                p.drawLine(QPointF(pa.sx, pa.sy), QPointF(pb.sx, pb.sy));
            }
        };
        drawAxis({1,0,0}, QColor(220,80,80));
        drawAxis({0,1,0}, QColor(80,220,80));
        drawAxis({0,0,1}, QColor(80,140,255));

        // ---- Trace polyline projected on y=0 ----
        if (trace_.size() > 1) {
            QPainterPath path;
            bool started = false;
            for (auto& pt : trace_) {
                auto pr = cam_.project(Vec3{pt.first, 0.02, pt.second});
                if (!pr.valid) { started = false; continue; }
                if (!started) { path.moveTo(pr.sx, pr.sy); started = true; }
                else path.lineTo(pr.sx, pr.sy);
            }
            p.setPen(QPen(QColor(255, 230, 80, 180), 1.5));
            p.drawPath(path);
        }

        // ---- Collect chassis + 4 wheel boxes into one face list ----
        std::vector<Face> faces;
        faces.reserve(48);

        auto* chassis = sim_.vehicle->chassis().get();
        Vec3 cAx = chassis->bodyToWorldDirection({1,0,0});
        Vec3 cAy = chassis->bodyToWorldDirection({0,1,0});
        Vec3 cAz = chassis->bodyToWorldDirection({0,0,1});
        const auto& he = sim_.vehicle->params().chassisHalfExtents;
        emitBox(faces, chassis->position, cAx, cAy, cAz, he,
                QColor(80, 130, 220), true);

        // Cabin / roof: smaller box on top, slightly forward.
        Vec3 cabinCenter{
            chassis->position.x + cAx.x * (-0.10) + cAy.x * (he.y + 0.20),
            chassis->position.y + cAx.y * (-0.10) + cAy.y * (he.y + 0.20),
            chassis->position.z + cAx.z * (-0.10) + cAy.z * (he.y + 0.20)
        };
        Vec3 cabinHE{ he.x * 0.55, 0.20, he.z * 0.85 };
        emitBox(faces, cabinCenter, cAx, cAy, cAz, cabinHE,
                QColor(60, 95, 170), true);

        for (int i = 0; i < 4; ++i) {
            const auto& w = sim_.vehicle->wheel(i);
            const RigidBody* wb = w.body().get();
            // Wheel-body axes: local +X = a radial dir (rotates with spin),
            // local +Y = spin axis (chassis -Z in world after attach),
            // local +Z = orthogonal radial.
            Vec3 wAx = wb->bodyToWorldDirection({1,0,0});
            Vec3 wAy = wb->bodyToWorldDirection({0,1,0});
            Vec3 wAz = wb->bodyToWorldDirection({0,0,1});
            double R = w.params().radius;
            double Wd = w.params().width * 0.5;
            // Half-extents along (local X, local Y=spin, local Z).
            Vec3 he2{R, Wd, R};
            QColor col = w.lastInContact() ? QColor(40,40,40) : QColor(180,60,60);
            emitBox(faces, wb->position, wAx, wAy, wAz, he2, col, true);
        }

        // Painter's algorithm: far → near.
        std::sort(faces.begin(), faces.end(),
                  [](const Face& a, const Face& b){ return a.depth > b.depth; });
        for (const auto& f : faces) {
            QPolygonF poly;
            for (int k = 0; k < f.n; ++k) poly << f.pts[k];
            p.setBrush(f.col);
            p.setPen(f.outline ? QPen(QColor(20,20,30), 1) : Qt::NoPen);
            p.drawPolygon(poly);
        }

        // Compass / camera info overlay.
        p.setPen(QColor(220, 230, 245));
        QFont ovf("Monospace", 9);
        p.setFont(ovf);
        p.drawText(r.x() + 10, r.y() + 18,
                   QString("cam az=%1°  el=%2°  d=%3 m   [LMB orbit | wheel zoom | 1/2/3/0 views]")
                       .arg(cam_.az * 180.0 / M_PI, 0, 'f', 0)
                       .arg(cam_.el * 180.0 / M_PI, 0, 'f', 0)
                       .arg(cam_.dist, 0, 'f', 1));

        p.restore();
    }

    [[maybe_unused]] void drawTopDown(QPainter& p, const QRect& r) {
        p.save();
        p.setClipRect(r);
        p.fillRect(r, QColor(40, 60, 45));

        // Camera: follow vehicle, +X right, +Z down (Z world axis).
        Vec3 c = sim_.vehicle->chassis()->position;
        const double scale = 25.0; // px per meter
        const double cx = r.x() + r.width() * 0.5;
        const double cy = r.y() + r.height() * 0.5;
        auto W2S = [&](double wx, double wz) {
            return QPointF(cx + (wx - c.x) * scale,
                           cy + (wz - c.z) * scale);
        };

        // Grid.
        p.setPen(QPen(QColor(70, 90, 75), 1));
        for (int gx = -50; gx <= 50; ++gx) {
            QPointF a = W2S(c.x + gx, c.z - 50);
            QPointF b = W2S(c.x + gx, c.z + 50);
            p.drawLine(a, b);
        }
        for (int gz = -50; gz <= 50; ++gz) {
            QPointF a = W2S(c.x - 50, c.z + gz);
            QPointF b = W2S(c.x + 50, c.z + gz);
            p.drawLine(a, b);
        }

        // Trace.
        if (trace_.size() > 1) {
            QPainterPath path;
            path.moveTo(W2S(trace_[0].first, trace_[0].second));
            for (size_t i = 1; i < trace_.size(); ++i)
                path.lineTo(W2S(trace_[i].first, trace_[i].second));
            p.setPen(QPen(QColor(255, 230, 80, 180), 1.5));
            p.drawPath(path);
        }

        // Vehicle chassis as a rotated rectangle.
        auto* chassis = sim_.vehicle->chassis().get();
        Vec3 fwd = chassis->bodyToWorldDirection({1.0, 0.0, 0.0});
        double yaw = std::atan2(-fwd.z, fwd.x); // top-down screen yaw
        const auto& he = sim_.vehicle->params().chassisHalfExtents;
        p.save();
        QPointF cs = W2S(c.x, c.z);
        p.translate(cs);
        // In top-down: X (forward) → screen +X, Z (right) → screen +Y. yaw rotates.
        p.rotate(-yaw * 180.0 / M_PI);
        QRectF body(-he.x * scale, -he.z * scale, 2*he.x * scale, 2*he.z * scale);
        p.setBrush(QColor(80, 130, 220, 220));
        p.setPen(QPen(QColor(220, 235, 255), 2));
        p.drawRect(body);
        // Heading arrow.
        p.setPen(QPen(QColor(255, 240, 100), 3));
        p.drawLine(QPointF(0, 0), QPointF(he.x * scale * 1.3, 0));
        p.restore();

        // Wheels.
        for (int i = 0; i < 4; ++i) {
            const auto& w = sim_.vehicle->wheel(i);
            Vec3 wp = w.body()->position;
            QPointF ws = W2S(wp.x, wp.z);
             // Top-down orientation comes from the KNUCKLE, not the wheel
            // body. The wheel body's local +X axis rotates about the spin
            // axis as the wheel rolls, which would make the rectangle
            // appear to "yaw" continuously even when the wheel is only
            // spinning. The knuckle's local +X is the true steering
            // forward direction (chassis-yaw + steer angle).
            const RigidBody* knk = w.knuckle();
            Vec3 wfwd = knk
                ? knk->bodyToWorldDirection({1.0, 0.0, 0.0})
                : w.body()->bodyToWorldDirection({1.0, 0.0, 0.0});
      
            double wyaw = std::atan2(-wfwd.z, wfwd.x);
            p.save();
            p.translate(ws);
            p.rotate(-wyaw * 180.0 / M_PI);
            double wL = w.params().radius * 2.0 * scale;
            double wW = w.params().width * scale;
            QColor col = w.lastInContact() ? QColor(40, 200, 60) : QColor(200, 60, 60);
            p.setBrush(col);
            p.setPen(QPen(Qt::white, 1.5));
            p.drawRect(QRectF(-wL * 0.5, -wW * 0.5, wL, wW));
            p.restore();
        }

        p.restore();
    }

    static QString fmt(double v, int decimals = 2) {
        return QString::number(v, 'f', decimals);
    }

    void drawHUD(QPainter& p, const QRect& r) {
        p.fillRect(r, QColor(22, 24, 30));
        p.setPen(QColor(220, 225, 240));
        QFont f("Monospace", 10);
        p.setFont(f);

        int y = r.y() + 18;
        const int x = r.x() + 14;
        auto line = [&](const QString& s) {
            p.drawText(x, y, s); y += 16;
        };

        line(QString("t = %1 s").arg(fmt(sim_.sys.getTime(), 2)));
        line(QString("v = %1 m/s  (%2 km/h)")
             .arg(fmt(sim_.vehicle->forwardSpeed()))
             .arg(fmt(sim_.vehicle->forwardSpeed() * 3.6, 1)));
        line(QString("vy_lat = %1 m/s").arg(fmt(sim_.vehicle->lateralSpeed())));
        line(QString("yaw = %1 deg")
             .arg(fmt(sim_.vehicle->yawAngle() * 180.0 / M_PI, 1)));
        line(QString("rpm  = %1")
             .arg(fmt(sim_.vehicle->drivetrain()->engineSpeed() * 60.0 / (2.0 * M_PI), 0)));
        line(QString("Teng = %1 N·m").arg(fmt(sim_.vehicle->drivetrain()->engineTorque())));
        y += 6;

        // Input bars.
        auto bar = [&](const QString& label, double val, double minV, double maxV,
                       QColor col) {
            p.setPen(QColor(200, 200, 220));
            p.drawText(x, y + 12, label);
            QRect bg(x + 100, y, r.width() - 130, 14);
            p.fillRect(bg, QColor(50, 50, 60));
            double t = (val - minV) / (maxV - minV);
            t = std::max(0.0, std::min(1.0, t));
            QRect fg(bg.x(), bg.y(), int(bg.width() * t), bg.height());
            p.fillRect(fg, col);
            p.setPen(QColor(220, 225, 240));
            p.drawText(bg, Qt::AlignCenter, fmt(val));
            y += 18;
        };
        bar("throttle", sim_.input.throttle, -1.0, 1.0, QColor(80, 200, 80));
        bar("brake",    sim_.input.brake,     0.0, 1.0, QColor(220, 70, 70));
        bar("steer",    sim_.input.steering, -1.0, 1.0, QColor(80, 160, 220));
        y += 6;

        // Per-wheel telemetry.
        const char* names[4] = {"FL", "FR", "RL", "RR"};
        line("Wheel  Fz[N]   κ      α[°]  spin[rpm]");
        for (int i = 0; i < 4; ++i) {
            const auto& w = sim_.vehicle->wheel(i);
            QString s = QString("%1   %2  %3  %4  %5")
                .arg(names[i], -3)
                .arg(fmt(w.lastFz(), 0), 6)
                .arg(fmt(w.lastKappa(), 3), 6)
                .arg(fmt(w.lastAlpha() * 180.0 / M_PI, 1), 5)
                .arg(fmt(w.lastSpin() * 60.0 / (2.0 * M_PI), 0), 6);
            line(s);
        }
        y += 6;

        line("Controls:");
        line("  W/S throttle / reverse");
        line("  A/D steer");
        line("  Space hand-brake");
        line("  R reset, P pause, Esc quit");
    }

private:
    VehicleSim sim_;
    QTimer* timer_ = nullptr;
    QElapsedTimer elapsed_;
    bool paused_ = false;
    std::deque<std::pair<double,double>> trace_;
    Camera3D cam_;
    QPoint   lastMouse_;
};

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    VehicleWindow win;
    win.show();
    return app.exec();
}
