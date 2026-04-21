/**
 * assembly_dynamics_qt.cpp
 * ============================================================
 * assets/assembly.step — OpenCASCADE tessellation + MBC++ dynamics
 *
 * OpenCASCADE ile assembly.step dosyasındaki her parçanın B-Rep
 * geometrisi triangulate edilir; MBC++ ile rijit-gövde dinamiği
 * çözülür; Qt5 ile gerçek-zamanlı 3D görselleştirme yapılır.
 *
 * Kontroller:
 *   Sol sürükle  – Döndür     Sağ sürükle – Kaydır
 *   Tekerlek     – Zoom        SPACE – Duraklat/Devam
 *   R – Sıfırla   +/- – Hız   W – Tel kafes    ESC – Çıkış
 */

// Qt
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

// STL
#include <cmath>
#include <limits>
#include <vector>
#include <array>
#include <string>
#include <deque>
#include <algorithm>
#include <memory>
#include <functional>

// OpenCASCADE
#include <STEPCAFControl_Reader.hxx>
#include <XCAFApp_Application.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>
#include <TDocStd_Document.hxx>
#include <TDF_LabelSequence.hxx>
#include <TDF_Label.hxx>
#include <TDataStd_Name.hxx>
#include <TCollection_AsciiString.hxx>
#include <TopLoc_Location.hxx>
#include <TopLoc_Datum3D.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS.hxx>
#include <TopExp_Explorer.hxx>
#include <BRep_Tool.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <Poly_Triangulation.hxx>
#include <gp_Pnt.hxx>
#include <gp_Trsf.hxx>
#include <Bnd_Box.hxx>
#include <BRepBndLib.hxx>

// MBC++
#include "mb/core/RigidBody.h"
#include "mb/constraints/RevoluteJoint.h"
#include "mb/constraints/SphericalJoint.h"
#include "mb/constraints/PrismaticJoint.h"
#include "mb/solvers/DirectSolver.h"
#include "mb/integrators/RungeKutta.h"
#include "mb/system/MultibodySystem.h"
#include "mb/core/ThreadConfig.h"

using namespace mb;

// ============================================================
//  OCCT helper: TopLoc_Location → gp_Trsf (compound compose)
// ============================================================
static gp_Trsf composeLoc(const TopLoc_Location& loc) {
    if (loc.IsIdentity()) return gp_Trsf();
    gp_Trsf t = loc.FirstDatum()->Transformation();
    if (loc.FirstPower() < 0) t.Invert();
    if (!loc.NextLocation().IsIdentity()) {
        gp_Trsf next = composeLoc(loc.NextLocation());
        t.Multiply(next);
    }
    return t;
}

// ============================================================
//  Veri yapıları
// ============================================================
struct Vec3f { float x, y, z; };
struct Tri   { int   a, b, c; };

struct PartMesh {
    std::string        name;
    std::vector<Vec3f> verts;  // dünya koordinatları (mm)
    std::vector<Tri>   tris;
    QColor             color;
    float cx = 0, cy = 0, cz = 0; // bounding-box merkezi (mm)
};

// Parça renk paleti (assembly.step sırasına göre, 11 parça)
static const QColor PALETTE[] = {
    QColor(210,155, 50),   // 0  crankshaft right
    QColor(255,185, 40),   // 1  crank pin
    QColor( 70,200, 90),   // 2  connecting rod
    QColor(100,175,255),   // 3  wrist pin
    QColor(175,200,220),   // 4  piston head
    QColor(200,150, 50),   // 5  crankshaft left
    QColor(230,170, 40),   // 6  Spur Gear 24t (2)
    QColor(200,140, 50),   // 7  PRIMARY DRIVE GEARS
    QColor(235,175, 35),   // 8  primary drive gear s
    QColor( 75,160,230),   // 9  Spur Gear 42t
    QColor(225,165, 55),   // 10 primary drive gear L
};

// ============================================================
//  STEP Yükleyici
// ============================================================
struct AssemblyLoader {
    std::vector<PartMesh> parts;
    float assemblyMin[3] = { 1e9f,  1e9f,  1e9f};
    float assemblyMax[3] = {-1e9f, -1e9f, -1e9f};

    bool load(const char* path) {
        // XDE uygulama & belge
        Handle(XCAFApp_Application) app = XCAFApp_Application::GetApplication();
        Handle(TDocStd_Document) doc;
        app->NewDocument("MDTV-XCAF", doc);

        STEPCAFControl_Reader reader;
        reader.SetNameMode(true);
        if (reader.ReadFile(path) != IFSelect_RetDone) {
            fprintf(stderr, "[STEP] ReadFile HATA: %s\n", path);
            return false;
        }
        reader.Transfer(doc);

        Handle(XCAFDoc_ShapeTool) ST =
            XCAFDoc_DocumentTool::ShapeTool(doc->Main());

        TDF_LabelSequence freeShapes;
        ST->GetFreeShapes(freeShapes);
        if (freeShapes.IsEmpty()) {
            fprintf(stderr, "[STEP] Serbest şekil bulunamadı\n");
            return false;
        }

        // Assembly kökü
        TDF_LabelSequence comps;
        ST->GetComponents(freeShapes.Value(1), comps, false);
        printf("[STEP] %d bileşen bulundu\n", comps.Length());

        int colorIdx = 0;
        for (int i = 1; i <= comps.Length(); i++) {
            TDF_Label comp = comps.Value(i);

            // İsim
            std::string name = "Part" + std::to_string(i);
            {
                Handle(TDataStd_Name) nm;
                if (comp.FindAttribute(TDataStd_Name::GetID(), nm)) {
                    TCollection_AsciiString s(nm->Get());
                    name = s.ToCString();
                }
            }

            // Şekli al (yerleşik konum dahil)
            TopoDS_Shape shapeLocated = ST->GetShape(comp);
            if (shapeLocated.IsNull()) { colorIdx++; continue; }

            // Dünya dönüşümü (mm)
            gp_Trsf worldTrsf = composeLoc(shapeLocated.Location());

            // Konumu sıfırla — triangulation shape-local koordinatlarda
            TopoDS_Shape shape = shapeLocated;
            shape.Location(TopLoc_Location()); // identity

            // Tessellate
            BRepMesh_IncrementalMesh mesher(shape, 0.25, false, 0.4);

            PartMesh pm;
            pm.name  = name;
            pm.color = PALETTE[colorIdx % (sizeof(PALETTE)/sizeof(PALETTE[0]))];

            for (TopExp_Explorer fe(shape, TopAbs_FACE); fe.More(); fe.Next()) {
                TopoDS_Face face = TopoDS::Face(fe.Current());
                TopLoc_Location fl;
                Handle(Poly_Triangulation) tr = BRep_Tool::Triangulation(face, fl);
                if (tr.IsNull()) continue;

                // face-local → shape-local → world
                gp_Trsf faceToWorld = worldTrsf;
                if (!fl.IsIdentity()) {
                    gp_Trsf ft = composeLoc(fl);
                    faceToWorld.Multiply(ft); // worldTrsf * fl
                    // Actually order matters: pt_world = worldTrsf * fl * pt_face
                    // gp_Trsf::Multiply means this = this * rhs, so:
                    // faceToWorld = worldTrsf first, then multiply by fl
                    // → faceToWorld = worldTrsf * fl ✓
                }

                bool rev = (face.Orientation() == TopAbs_REVERSED);
                int  base = (int)pm.verts.size();

                for (int n = 1; n <= tr->NbNodes(); n++) {
                    gp_Pnt pt = tr->Node(n);
                    pt.Transform(faceToWorld);
                    pm.verts.push_back({(float)pt.X(), (float)pt.Y(), (float)pt.Z()});
                }
                for (int t = 1; t <= tr->NbTriangles(); t++) {
                    int n1, n2, n3;
                    tr->Triangle(t).Get(n1, n2, n3);
                    if (rev) std::swap(n2, n3);
                    pm.tris.push_back({base+n1-1, base+n2-1, base+n3-1});
                }
            }

            if (pm.verts.empty()) { colorIdx++; continue; }

            // Bounding box merkezi
            float sx=0,sy=0,sz=0;
            for (auto& v : pm.verts) {
                sx+=v.x; sy+=v.y; sz+=v.z;
                assemblyMin[0]=std::min(assemblyMin[0],v.x);
                assemblyMin[1]=std::min(assemblyMin[1],v.y);
                assemblyMin[2]=std::min(assemblyMin[2],v.z);
                assemblyMax[0]=std::max(assemblyMax[0],v.x);
                assemblyMax[1]=std::max(assemblyMax[1],v.y);
                assemblyMax[2]=std::max(assemblyMax[2],v.z);
            }
            float fn = (float)pm.verts.size();
            pm.cx=sx/fn; pm.cy=sy/fn; pm.cz=sz/fn;

            printf("  [%2d] %-35s  verts=%-6d tris=%-5d  c=(%.1f,%.1f,%.1f)mm\n",
                   (int)parts.size(), name.c_str(),
                   (int)pm.verts.size(), (int)pm.tris.size(),
                   pm.cx, pm.cy, pm.cz);

            parts.push_back(std::move(pm));
            colorIdx++;
        }
        return !parts.empty();
    }
};

// ============================================================
//  Kamera
// ============================================================
struct Camera {
    double az=0.5, el=0.35, zoom=1.0;
    double panX=0, panY=0;
    int W=1280, H=800;

    Vec3f right()   const { return {(float) cos(az), 0.f, (float)-sin(az)}; }
    Vec3f up()      const { return {(float)(sin(az)*sin(el)),(float)cos(el),(float)(cos(az)*sin(el))}; }
    Vec3f forward() const { return {(float)(sin(az)*cos(el)),(float)-sin(el),(float)(cos(az)*cos(el))}; }

    QPointF project(float x, float y, float z) const {
        Vec3f r=right(), u=up();
        double sx = (r.x*x+r.y*y+r.z*z)*zoom + panX;
        double sy = -(u.x*x+u.y*y+u.z*z)*zoom + panY;
        return {sx+W*0.5, sy+H*0.5};
    }
    float depth(float x, float y, float z) const {
        Vec3f f=forward();
        return f.x*x+f.y*y+f.z*z;
    }
};

// ============================================================
//  MBC++ Simülasyonu
// ============================================================
struct Simulation {
    MultibodySystem sys{"AssemblyDyn"};
    std::shared_ptr<RigidBody> ground, crank, rod, piston, drivenGear;

    // Fiziksel parametreler (metre) — assembly.step'ten ölçüldü:
    //   crank pin Y=35.8mm, piston Y=140.6mm → R=35.8mm, L=104.8mm
    double R = 0.0358;  // krank yarıçapı = 35.8mm
    double L = 0.1048;  // biyel uzunluğu = 104.8mm
    double OMEGA = 1500.0*2.0*M_PI/60.0; // 1500 rpm başlangıç
    double speedFactor = 1.0;
    double dt = 0.00005;
    int    subSteps = 6;

    // Parça referans merkezleri (mm, STEP'ten — her parçanın yerel sıfırı)
    // Bunlar loader'dan alınır, dinamik dönüşüm sırasında yerel merkez
    // olarak kullanılır.
    std::array<float,3> partCenter[11] = {};

    // Simülasyon anlık transform'ları — render için
    // v_world = R * (v_step - pivot) + (tx,ty,tz)
    // pivot: krank grubunda (0,0,0) = STEP origin; diğerlerinde part kendi merkezi (NaN)
    struct XForm {
        double tx,ty,tz;
        double R[9];
        double pivX = std::numeric_limits<double>::quiet_NaN();
        double pivY = std::numeric_limits<double>::quiet_NaN();
        double pivZ = std::numeric_limits<double>::quiet_NaN();
    } xf[11];

    // Piston geçmişi
    struct Sample { double t, y; };
    std::deque<Sample> history;

    void build() {
        sys = MultibodySystem("AssemblyDyn");
        sys.setGravity(Vec3(0,-9.81,0));

        ground = RigidBody::createGround("Ground");
        sys.addBody(ground);

        // ── Tutarlı başlangıç koşulları ──────────────────────────────
        // Krank X ekseni etrafında döner; pin gövde çerçevesinde (0,R,0).
        // θ₀ = π/2 seçildi: en sade tutarsızlıksız başlangıç noktası.
        //
        //  pin_dünya = Rx(π/2)·(0,R,0) = (0, 0, R)
        //  v_pin     = ω×r = (Ω,0,0)×(0,0,R) = (0,-ΩR,0)   ← saf Y hızı
        //  y_piston  = √(L²-R²)                              ← piston konumu
        //  v_piston  = -ΩR  (Y ekseni boyunca)              ← aynı hız → biyel ω=0
        //
        // Böylece başlangıçta tüm hız kısıtları otomatik sağlanır.
        // ────────────────────────────────────────────────────────────

        const double theta0   = M_PI / 2.0;
        const double omegaEff = OMEGA * speedFactor;
        const double pinZ     = R;                       // pin_dünya = (0,0,R)
        const double pisY     = std::sqrt(L*L - R*R);   // piston Y konumu
        const double vPisY    = -omegaEff * R;           // piston hızı (Y)

        // Rod yönü: pin(0,0,R) → piston(0,pisY,0)
        // Gövde-Y eksenini bu doğrultuya çevirmek için X etrafında döndür:
        // (0,1,0) → (0,pisY/L,-R/L)  ⇒  α = atan2(-R/L, pisY/L)
        const double rodAngle = std::atan2(-R / L, pisY / L);

        // Krank: Rx(π/2)
        crank = RigidBody::createCylinder(0.4, R, 0.120, "Crank");
        crank->position        = Vec3(0,0,0);
        crank->orientation     = Quaternion::fromAxisAngle(Vec3(1,0,0), theta0);
        crank->angularVelocity = Vec3(omegaEff, 0, 0);
        sys.addBody(crank);

        // Biyel: orta nokta = ortalaması(pin, piston)
        rod = RigidBody::createRod(0.06, L, 0.004, "Rod");
        rod->position        = Vec3(0, pisY * 0.5, pinZ * 0.5);
        rod->orientation     = Quaternion::fromAxisAngle(Vec3(1,0,0), rodAngle);
        rod->velocity        = Vec3(0, vPisY, 0); // pin ve piston aynı hızda → ω_rod=0
        rod->angularVelocity = Vec3(0, 0, 0);
        sys.addBody(rod);

        // Piston
        piston = RigidBody::createCylinder(0.15, 0.019, 0.040, "Piston");
        piston->position = Vec3(0, pisY, 0);
        piston->velocity = Vec3(0, vPisY, 0);
        sys.addBody(piston);

        // Çıkış dişlisi — STEP'te X=83mm, Y≈-2mm; X ekseni etrafında döner
        double gdX = 0.083;
        drivenGear = RigidBody::createCylinder(0.12, 0.042, 0.012, "DrivenGear");
        drivenGear->position       = Vec3(gdX, 0, 0);
        drivenGear->orientation    = Quaternion::identity();
        drivenGear->angularVelocity = Vec3(-OMEGA*speedFactor*(24.0/42.0), 0, 0);
        sys.addBody(drivenGear);

        // Eklemler
        // Krank: X ekseni etrafında döner
        sys.addConstraint(std::make_shared<RevoluteJoint>(
            ground.get(), crank.get(),
            Vec3(0,0,0), Vec3(0,0,0),
            Vec3(1,0,0), Vec3(1,0,0), 0.0, "J_Crank"));
        // Krank pimi: krank gövdesinde (0, R, 0) noktası — X'e dik, Y'ye offset
        sys.addConstraint(std::make_shared<SphericalJoint>(
            crank.get(), rod.get(),
            Vec3(0,R,0), Vec3(0,-L*0.5,0), "J_RodCrank"));
        sys.addConstraint(std::make_shared<SphericalJoint>(
            rod.get(), piston.get(),
            Vec3(0,L*0.5,0), Vec3(0,0,0), "J_RodPiston"));
        sys.addConstraint(std::make_shared<PrismaticJoint>(
            ground.get(), piston.get(),
            Vec3(0,0,0), Vec3(0,0,0),
            Vec3(0,1,0), Vec3(0,1,0), "J_Slide"));
        sys.addConstraint(std::make_shared<RevoluteJoint>(
            ground.get(), drivenGear.get(),
            Vec3(gdX,0,0), Vec3(0,0,0),
            Vec3(1,0,0), Vec3(1,0,0), 0.0, "J_Driven"));

        sys.setSolver(std::make_shared<DirectSolver>());
        IntegratorConfig cfg;
        cfg.adaptive=true; cfg.absTol=1e-6; cfg.relTol=1e-5;
        cfg.maxStep=0.001; cfg.minStep=1e-8;
        sys.setIntegrator(std::make_shared<DormandPrince45>(cfg));
        sys.initialize();
        history.clear();
        updateXForms();
    }

    void step() {
        for (int i=0;i<subSteps;i++) sys.step(dt);
        updateXForms();
        history.push_back({sys.getTime(), piston->position.y});
        if ((int)history.size() > 800) history.pop_front();
    }

    // Quaternion → satır-ana 3×3
    static void qToR(const Quaternion& q, double R[9]) {
        double w=q.w,x=q.x,y=q.y,z=q.z;
        R[0]=1-2*(y*y+z*z); R[1]=2*(x*y-w*z); R[2]=2*(x*z+w*y);
        R[3]=2*(x*y+w*z); R[4]=1-2*(x*x+z*z); R[5]=2*(y*z-w*x);
        R[6]=2*(x*z-w*y); R[7]=2*(y*z+w*x); R[8]=1-2*(x*x+y*y);
    }

    void setXF(int i, double tx,double ty,double tz, const double R[9]) {
        if (i<0||i>=11) return;
        xf[i].tx=tx; xf[i].ty=ty; xf[i].tz=tz;
        for(int k=0;k<9;k++) xf[i].R[k]=R[k];
    }

    void updateXForms() {
        const double S=1e3; // m→mm
        double Rc[9]; qToR(crank->orientation,Rc);
        double Rr[9]; qToR(rod->orientation,Rr);
        double Rp[9]; qToR(piston->orientation,Rp);
        double Rd[9]; qToR(drivenGear->orientation,Rd);

        // ── Krank grubu: pivot = STEP krank dönme ekseni ≈ (0,0,0) ──────────
        // v_world = R_crank * v_step  (X offset korunur, YZ döner)
        // Krank gövdesi (0,5), krank pimi (1), drive gearler (6,7,8)
        auto setCrankGroup = [&](int i) {
            setXF(i, 0,0,0, Rc);
            xf[i].pivX=0; xf[i].pivY=0; xf[i].pivZ=0;
        };
        setCrankGroup(0); // crankshaft right
        setCrankGroup(1); // crank pin
        setCrankGroup(5); // crankshaft left
        setCrankGroup(6); // PRIMARY DRIVE GEARS  (X≈54mm)
        setCrankGroup(7); // primary drive gear s (X≈84mm)
        setCrankGroup(8); // primary drive gear L (X≈107mm)

        // ── Biyel (2): kendi merkezi etrafında döner + sim pozisyonu ──────────
        setXF(2, rod->position.x*S, rod->position.y*S, rod->position.z*S, Rr);

        // ── Piston grubu (3,4): sadece Y boyunca öteleme ─────────────────────
        setXF(3, piston->position.x*S, piston->position.y*S, piston->position.z*S, Rp);
        setXF(4, piston->position.x*S, piston->position.y*S, piston->position.z*S, Rp);

        // ── Çıkış dişlisi (9,10) — 9 parça olduğundan hiç kullanılmıyor ──────
        setXF(9,  drivenGear->position.x*S, drivenGear->position.y*S, drivenGear->position.z*S, Rd);
        setXF(10, drivenGear->position.x*S, drivenGear->position.y*S, drivenGear->position.z*S, Rd);
    }

    double rpm() const {
        return std::abs(crank->angularVelocity.x)*60.0/(2.0*M_PI);
    }
};

// ============================================================
//  Qt Widget
// ============================================================
class AssemblyWidget : public QWidget {
    Q_OBJECT
public:
    explicit AssemblyWidget(QWidget* parent=nullptr): QWidget(parent) {
        setWindowTitle("MBC++ — assembly.step | 3D Dinamik Analiz");
        resize(1280,800);
        setMinimumSize(900,600);
        setFocusPolicy(Qt::StrongFocus);

        // STEP yükle
        printf("[STEP] Yükleniyor...\n");
        loader_.load("/home/burak/MBC++/assets/assembly.step");

        // Kamera ölçeği: assembly bounding box'a göre
        float spanX = loader_.assemblyMax[0]-loader_.assemblyMin[0];
        float spanY = loader_.assemblyMax[1]-loader_.assemblyMin[1];
        float spanZ = loader_.assemblyMax[2]-loader_.assemblyMin[2];
        float span  = std::max({spanX,spanY,spanZ,1.0f});
        cam_.zoom   = (float)std::min(width(),height())*0.50 / span;

        // Kamera merkezi
        float cx=(loader_.assemblyMin[0]+loader_.assemblyMax[0])*0.5f;
        float cy=(loader_.assemblyMin[1]+loader_.assemblyMax[1])*0.5f;
        cam_.panX = -(cx * cam_.zoom);
        cam_.panY =  (cy * cam_.zoom);

        // Parça merkezlerini simülasyona aktar
        for (int i=0; i<(int)loader_.parts.size() && i<11; i++) {
            sim_.partCenter[i] = {loader_.parts[i].cx,
                                  loader_.parts[i].cy,
                                  loader_.parts[i].cz};
        }
        sim_.build();

        timer_ = new QTimer(this);
        connect(timer_,&QTimer::timeout,this,&AssemblyWidget::tick);
        timer_->start(16);
        elapsed_.start();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        cam_.W=width(); cam_.H=height();
        p.fillRect(rect(), QColor(8,10,16));
        drawGrid(p);
        renderParts(p);
        drawGraph(p);
        drawHUD(p);
    }
    void keyPressEvent(QKeyEvent* e) override {
        switch(e->key()) {
        case Qt::Key_Space: paused_=!paused_; break;
        case Qt::Key_R: resetSim(); break;
        case Qt::Key_Plus: case Qt::Key_Equal:
            sim_.speedFactor=std::min(sim_.speedFactor*1.5,8.0); break;
        case Qt::Key_Minus:
            sim_.speedFactor=std::max(sim_.speedFactor/1.5,0.1); break;
        case Qt::Key_W: wireframe_=!wireframe_; break;
        case Qt::Key_Escape: close(); break;
        }
        update();
    }
    void mousePressEvent(QMouseEvent* e)  override { lastMouse_=e->pos(); }
    void mouseMoveEvent(QMouseEvent* e)   override {
        QPoint d=e->pos()-lastMouse_; lastMouse_=e->pos();
        if (e->buttons()&Qt::LeftButton) {
            cam_.az-=d.x()*0.006;
            cam_.el =std::max(-1.4,std::min(1.4,cam_.el-d.y()*0.006));
        }
        if (e->buttons()&(Qt::RightButton|Qt::MiddleButton)) {
            cam_.panX+=d.x(); cam_.panY+=d.y();
        }
        update();
    }
    void wheelEvent(QWheelEvent* e) override {
        cam_.zoom *= (e->angleDelta().y()>0)?1.12:1.0/1.12;
        update();
    }
    void resizeEvent(QResizeEvent*) override { cam_.W=width(); cam_.H=height(); }

private slots:
    void tick() {
        if (!paused_) sim_.step();
        frameCount_++;
        update();
    }

private:
    AssemblyLoader loader_;
    Simulation     sim_;
    Camera         cam_;
    QTimer*        timer_=nullptr;
    QElapsedTimer  elapsed_;
    bool           paused_=false, wireframe_=false;
    int            frameCount_=0;
    QPoint         lastMouse_;

    void resetSim() {
        sim_.build();
        elapsed_.restart(); frameCount_=0;
        // Kamera sıfırla
        cam_.az=0.5; cam_.el=0.35;
        float span=std::max({loader_.assemblyMax[0]-loader_.assemblyMin[0],
                             loader_.assemblyMax[1]-loader_.assemblyMin[1],
                             loader_.assemblyMax[2]-loader_.assemblyMin[2], 1.0f});
        cam_.zoom=(float)std::min(width(),height())*0.50/span;
        float cx=(loader_.assemblyMin[0]+loader_.assemblyMax[0])*0.5f;
        float cy=(loader_.assemblyMin[1]+loader_.assemblyMax[1])*0.5f;
        cam_.panX=-(cx*cam_.zoom); cam_.panY=(cy*cam_.zoom);
    }

    // Zemin ızgarası
    void drawGrid(QPainter& p) {
        float yf = loader_.assemblyMin[1]-8.0f;
        p.setPen(QPen(QColor(20,25,38),1));
        float step=20.0f, ext=220.0f;
        for (float x=-ext;x<=ext;x+=step) {
            p.drawLine(cam_.project(x,yf,-ext),cam_.project(x,yf,ext));
        }
        for (float z=-ext;z<=ext;z+=step) {
            p.drawLine(cam_.project(-ext,yf,z),cam_.project(ext,yf,z));
        }
        // Eksenler
        p.setPen(QPen(QColor(80,0,0),1.5));
        p.drawLine(cam_.project(0,yf,0),cam_.project(50,yf,0));
        p.setPen(QPen(QColor(0,80,0),1.5));
        p.drawLine(cam_.project(0,yf,0),cam_.project(0,yf+50,0));
        p.setPen(QPen(QColor(0,0,80),1.5));
        p.drawLine(cam_.project(0,yf,0),cam_.project(0,yf,50));
    }

    // Bir parçayı çiz — xformIdx: sim_.xf[] dizisi indeksi
    void drawPart(QPainter& p, int partIdx, int xformIdx) {
        if (partIdx>=(int)loader_.parts.size()) return;
        const PartMesh& pm = loader_.parts[partIdx];
        if (pm.verts.empty()) return;
        const Simulation::XForm& xf = sim_.xf[xformIdx];

        // v_world = R * (v_step - pivot) + (tx,ty,tz)
        // pivot: krank grubunda (0,0,0); diğerlerinde parça kendi merkezi
        const bool hasPiv = !std::isnan(xf.pivX);
        float pcx = hasPiv ? (float)xf.pivX : pm.cx;
        float pcy = hasPiv ? (float)xf.pivY : pm.cy;
        float pcz = hasPiv ? (float)xf.pivZ : pm.cz;
        const double* R = xf.R;
        auto transform = [&](const Vec3f& v) -> Vec3f {
            float lx=v.x-pcx, ly=v.y-pcy, lz=v.z-pcz;
            return {
                (float)(R[0]*lx+R[1]*ly+R[2]*lz + xf.tx),
                (float)(R[3]*lx+R[4]*ly+R[5]*lz + xf.ty),
                (float)(R[6]*lx+R[7]*ly+R[8]*lz + xf.tz)
            };
        };

        Vec3f fwd = cam_.forward();

        if (wireframe_) {
            p.setPen(QPen(pm.color.darker(140),0.5));
            for (auto& t : pm.tris) {
                auto a=transform(pm.verts[t.a]);
                auto b=transform(pm.verts[t.b]);
                auto c=transform(pm.verts[t.c]);
                p.drawLine(cam_.project(a.x,a.y,a.z),cam_.project(b.x,b.y,b.z));
                p.drawLine(cam_.project(b.x,b.y,b.z),cam_.project(c.x,c.y,c.z));
                p.drawLine(cam_.project(c.x,c.y,c.z),cam_.project(a.x,a.y,a.z));
            }
            return;
        }

        // Katı render: depth sort + flat shading
        struct Face {
            float   depth;
            QPointF pts[3];
            float   nx,ny,nz;
        };
        std::vector<Face> faces;
        faces.reserve(pm.tris.size());

        for (auto& t : pm.tris) {
            auto a=transform(pm.verts[t.a]);
            auto b=transform(pm.verts[t.b]);
            auto c=transform(pm.verts[t.c]);

            float abx=b.x-a.x,aby=b.y-a.y,abz=b.z-a.z;
            float acx=c.x-a.x,acy=c.y-a.y,acz=c.z-a.z;
            float nx=aby*acz-abz*acy;
            float ny=abz*acx-abx*acz;
            float nz=abx*acy-aby*acx;
            float nl=std::sqrt(nx*nx+ny*ny+nz*nz)+1e-12f;
            nx/=nl; ny/=nl; nz/=nl;

            // Hafif back-face culling
            if (nx*fwd.x+ny*fwd.y+nz*fwd.z > 0.12f) continue;

            float mx=(a.x+b.x+c.x)/3, my=(a.y+b.y+c.y)/3, mz=(a.z+b.z+c.z)/3;
            Face f;
            f.depth = cam_.depth(mx,my,mz);
            f.pts[0]=cam_.project(a.x,a.y,a.z);
            f.pts[1]=cam_.project(b.x,b.y,b.z);
            f.pts[2]=cam_.project(c.x,c.y,c.z);
            f.nx=nx; f.ny=ny; f.nz=nz;
            faces.push_back(f);
        }

        std::sort(faces.begin(),faces.end(),[](const Face& a,const Face& b){return a.depth<b.depth;});

        const float LX=0.35f, LY=0.75f, LZ=0.55f;
        const float LL=std::sqrt(LX*LX+LY*LY+LZ*LZ);

        for (auto& f : faces) {
            float d = std::max(0.0f, f.nx*LX/LL+f.ny*LY/LL+f.nz*LZ/LL);
            float b = 0.22f+(1.0f-0.22f)*d;
            QColor col(
                std::min(255,(int)(pm.color.red()*b)),
                std::min(255,(int)(pm.color.green()*b)),
                std::min(255,(int)(pm.color.blue()*b)),
                215);
            QPainterPath path;
            path.moveTo(f.pts[0]); path.lineTo(f.pts[1]);
            path.lineTo(f.pts[2]); path.closeSubpath();
            p.fillPath(path,col);
            p.setPen(QPen(QColor(0,0,0,18),0.3));
            p.drawPath(path);
        }
    }

    // Tüm parçaları depth-sorted olarak çiz
    void renderParts(QPainter& p) {
        int n = std::min((int)loader_.parts.size(), 11);
        struct Ent { int pi, xi; float d; };
        std::vector<Ent> order;
        for (int i=0;i<n;i++) {
            const Simulation::XForm& xf=sim_.xf[i];
            order.push_back({i,i,cam_.depth((float)xf.tx,(float)xf.ty,(float)xf.tz)});
        }
        std::sort(order.begin(),order.end(),[](const Ent& a,const Ent& b){return a.d<b.d;});
        for (auto& e:order) drawPart(p,e.pi,e.xi);
    }

    // Piston Y grafiği
    void drawGraph(QPainter& p) {
        if (sim_.history.size()<2) return;
        const double S=1e3;
        int gx=width()-355,gy=height()-205,gw=325,gh=175;
        p.fillRect(gx,gy,gw,gh,QColor(10,12,22,210));
        p.setPen(QPen(QColor(30,35,52),1)); p.drawRect(gx,gy,gw,gh);
        p.setFont(QFont("Sans",9,QFont::Bold));
        p.setPen(QColor(155,160,175));
        p.drawText(gx+8,gy+15,"Piston Y (mm)");

        int px=gx+48,py=gy+25,pw=gw-60,ph=gh-42;
        double ymn=1e9,ymx=-1e9;
        for (auto& s:sim_.history){double y=s.y*S; ymn=std::min(ymn,y); ymx=std::max(ymx,y);}
        if (ymx-ymn<1.0){ymn-=1;ymx+=1;}
        double t0=sim_.history.front().t,t1=sim_.history.back().t;
        if (t1-t0<1e-6) return;

        p.setPen(QPen(QColor(28,32,48),1));
        for(int i=0;i<=4;i++) p.drawLine(px,py+i*ph/4,px+pw,py+i*ph/4);
        p.setFont(QFont("Monospace",7));
        p.setPen(QColor(100,105,120));
        p.drawText(px-44,py+4,   QString("%1").arg(ymx,0,'f',0));
        p.drawText(px-44,py+ph+4,QString("%1").arg(ymn,0,'f',0));
        p.drawText(px-14,py+ph+15,"mm");

        p.setPen(QPen(QColor(75,195,255),1.6));
        QPointF prev; bool first=true;
        double ts=t1-t0;
        for (auto& s:sim_.history) {
            double fx=(s.t-t0)/ts;
            double fy=(s.y*S-ymn)/(ymx-ymn);
            fy=std::max(0.0,std::min(1.0,fy));
            QPointF pt(px+fx*pw, py+ph-fy*ph);
            if(!first) p.drawLine(prev,pt);
            prev=pt; first=false;
        }
        double cur=sim_.history.back().y*S;
        p.setFont(QFont("Monospace",9,QFont::Bold));
        p.setPen(QColor(75,195,255));
        p.drawText(px+pw-88,py-2,QString("%1 mm").arg(cur,0,'f',2));
    }

    // HUD
    void drawHUD(QPainter& p) {
        double fps = frameCount_/(elapsed_.elapsed()*0.001+1e-9);
        auto a = sim_.sys.analyze();

        p.setFont(QFont("Monospace",11));
        p.setPen(QColor(205,210,225));
        int x=16,y=28,dy=20;
        auto L=[&](const QString& s){p.drawText(x,y,s);y+=dy;};
        L(QString("t   = %1 s").arg(sim_.sys.getTime(),0,'f',4));
        L(QString("FPS = %1").arg(fps,0,'f',1));
        L(QString("RPM = %1").arg(sim_.rpm(),0,'f',0));
        L(QString("Hız × %1").arg(sim_.speedFactor,0,'f',1));
        L("");
        p.setPen(QColor(170,195,235)); p.setFont(QFont("Monospace",10));
        L(QString("KE  = %1 J").arg(a.kineticEnergy,0,'f',5));
        L(QString("PE  = %1 J").arg(a.potentialEnergy,0,'f',5));
        p.setPen(QColor(255,110,70));
        L(QString("|C| = %1").arg(a.constraintViolation,0,'e',2));
        y+=4;
        p.setPen(QColor(130,195,130)); p.setFont(QFont("Monospace",10));
        L(QString("Piston Y : %1 mm").arg(sim_.piston->position.y*1e3,0,'f',2));
        L(QString("Parçalar : %1").arg((int)loader_.parts.size()));

        // Sağ panel
        int bx=width()-240,by=28;
        p.setFont(QFont("Sans",9)); p.setPen(QColor(100,105,130));
        auto RL=[&](const QString& s){p.drawText(bx,by,s);by+=18;};
        RL("Sol sürükle   Döndür");
        RL("Sağ sürükle   Kaydır");
        RL("Tekerlek      Zoom");
        RL("SPACE         Duraklat");
        RL("R             Sıfırla");
        RL("+/-           Hız");
        RL("W             Tel kafes");
        RL("ESC           Çıkış");
        by+=6;
        for (int i=0;i<(int)loader_.parts.size()&&i<11;i++) {
            p.setPen(loader_.parts[i].color);
            RL(QString("■ %1").arg(QString::fromStdString(loader_.parts[i].name).left(27)));
        }

        if (paused_) {
            p.setPen(QColor(255,80,80));
            p.setFont(QFont("Sans",18,QFont::Bold));
            p.drawText(width()/2-70,46,"⏸  DURAKLATILDI");
        }
        if (wireframe_) {
            p.setPen(QColor(255,195,60));
            p.setFont(QFont("Sans",10,QFont::Bold));
            p.drawText(width()/2-48,height()-18,"[ TEL KAFES ]");
        }
    }
};

// ============================================================
int main(int argc, char* argv[]) {
    for (int i=1;i<argc;i++)
        if (std::string(argv[i])=="-c"&&i+1<argc)
            ThreadConfig::setNumThreads(std::atoi(argv[++i]));
    printf("[MBC++] Assembly 3D Dinamik | %d thread(s)\n", ThreadConfig::numThreads());
    QApplication app(argc,argv);
    AssemblyWidget win;
    win.show();
    return app.exec();
}

#include "assembly_dynamics_qt.moc"
