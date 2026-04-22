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
#include <QProcess>
#include <QImage>

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
#include "mb/forces/AppliedForce.h"
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

            printf("  [%2d] %-35s  verts=%-6d tris=%-5d  c=(%.6f, %.6f, %.6f)mm\n",
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

    // Fiziksel parametreler (metre) 
    double R = 0.0379515;   // True 3D hypotenuse based on pure geometric axis
    double L = 0.0966523;   // Wrist pin Y(132.43mm) - Krank pimi
    static constexpr double PISTON_JOINT_Y = -0.008134;
    
    double OMEGA = 1500.0*2.0*M_PI/60.0; // 1500 rpm başlangıç

    // Gaz kuvveti parametreleri (sabit)
    static constexpr double BORE_RADIUS = 0.019;  // 19 mm piston çapı
    static constexpr double A_BORE = M_PI * BORE_RADIUS * BORE_RADIUS; // ≈1.134e-3 m²
    static constexpr double P_ATM  = 1.0e5;   // 1 bar atmosfer [Pa]
    static constexpr double P_PEAK = 60.0e5;  // 60 bar tepe yanma [Pa]
    double speedFactor = 1.0;
    double dt = 0.00004;   // sabit adım — adaptif integratör değişkeni değil
    int    subSteps = 8;   // frame başı adım sayısı

    // Parça referans merkezleri (mm, STEP'ten — her parçanın yerel sıfırı)
    // Bunlar loader'dan alınır, dinamik dönüşüm sırasında yerel merkez
    // olarak kullanılır.
    std::array<float,3> partCenter[11] = {};
    float rodBigEndY   = 35.8f;   // STEP'ten ölçülen gerçek big-end Y (mm)
    float rodSmallEndY = 132.4f;  // STEP'ten ölçülen gerçek small-end Y (mm)

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

        // ── Tutarlı başlangıç koşulları — ÜÖN (TDC, θ=0) ───────────────────
        // θ₀=0: krank pimi en üstte (0,R,0). Gaz kuvveti (cosθ)⁴ θ=0'da
        // maksimum olduğundan piston anında aşağı itilir → sistem kilitsiz.
        //
        // θ=0 tutarlı hız türetimi (krank ω_x = +Ω, X ekseni sağ-el):
        //   pin_dünya  = (0, R, 0)
        //   v_pin      = (Ω,0,0)×(0,R,0) = (0, 0, ΩR)       ← saf Z hızı
        //   piston_Y   = R + L   (TDC, maksimum konum)
        //   v_piston   = 0       (dönüm noktası)
        //   rod ω_x    = -ΩR/L  (kısıt türetmesinden)
        //   rod v_cm   = (0, 0, ΩR/2)
        // ────────────────────────────────────────────────────────────

        const double omegaEff  = OMEGA * speedFactor;
        // TDC: bilek pimi Y = R+L = 132.4mm, piston kafası merkezi = R+L−PISTON_JOINT_Y = 140.6mm
        const double wristY    = R + L;              // bilek pimi konumu (eklem noktası)
        const double pisY      = wristY - PISTON_JOINT_Y; // piston STEP merkezi (140.6mm)
        const double rodOmegaX = -omegaEff * R / L; // biyel açısal hızı
        const double rodVZ     =  omegaEff * R / 2.0; // biyel merkezi Z hızı

        // Krank: Rx(0) = birim — pin gövde çerçevesinde (0,R,0) → dünya (0,R,0)
        crank = RigidBody::createCylinder(1.5, R, 0.120, "Crank"); // 1.5 kg volan
        crank->position        = Vec3(0,0,0);
        crank->orientation     = Quaternion::identity();
        crank->angularVelocity = Vec3(omegaEff, 0, 0);
        sys.addBody(crank);

        // Biyel: pin(0,R,0) → piston(0,R+L,0), gövde-Y = dünya-Y → orient=I
        rod = RigidBody::createRod(0.06, L, 0.004, "Rod");
        rod->position        = Vec3(0, R + L * 0.5, 0);
        rod->orientation     = Quaternion::identity();
        rod->velocity        = Vec3(0, 0, rodVZ);
        rod->angularVelocity = Vec3(rodOmegaX, 0, 0);
        sys.addBody(rod);

        // Piston: TDC'de hareketsiz
        piston = RigidBody::createCylinder(0.30, 0.019, 0.040, "Piston"); // 300 g
        piston->position = Vec3(0, pisY, 0);
        piston->velocity = Vec3(0, 0, 0);
        sys.addBody(piston);

        // Çıkış dişlisi — STEP'te X=83mm, Y≈-2mm; X ekseni etrafında döner
        double gdX = 0.083;
        drivenGear = RigidBody::createCylinder(0.12, 0.042, 0.012, "DrivenGear");
        drivenGear->position       = Vec3(gdX, 0, 0);
        drivenGear->orientation    = Quaternion::identity();
        drivenGear->angularVelocity = Vec3(-OMEGA*speedFactor*(24.0/42.0), 0, 0);
        sys.addBody(drivenGear);

        // Eklemler
        // Krank: X ekseni etrafında döner — yatak sönümü ile yük direnci
        auto jCrank = std::make_shared<RevoluteJoint>(
            ground.get(), crank.get(),
            Vec3(0,0,0), Vec3(0,0,0),
            Vec3(1,0,0), Vec3(1,0,0), 0.0, "J_Crank");
        jCrank->setDamping(0.05); // 0.05 N·m·s — hafif yatak sürtünmesi
        sys.addConstraint(jCrank);
        // Krank pimi: krank gövdesinde (0, R, 0) noktası — X'e dik, Y'ye offset
        sys.addConstraint(std::make_shared<SphericalJoint>(
            crank.get(), rod.get(),
            Vec3(0,R,0), Vec3(0,-L*0.5,0), "J_RodCrank"));
        // Bilek pimi: biyel ucunda (0,L/2,0), piston gövdesinde (0,PISTON_JOINT_Y,0)
        sys.addConstraint(std::make_shared<SphericalJoint>(
            rod.get(), piston.get(),
            Vec3(0,L*0.5,0), Vec3(0,PISTON_JOINT_Y,0), "J_RodPiston"));
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
        cfg.adaptive=true; cfg.absTol=1e-7; cfg.relTol=1e-6;
        cfg.maxStep=0.0002; cfg.minStep=1e-8;
        sys.setIntegrator(std::make_shared<DormandPrince45>(cfg));

        // ── Piston force — applied directly to the piston ─────────────────
        // The combustion force is applied at the wrist pin location.
        // The MBS solver automatically converts this to crank torque 
        // through the connecting rod and joints.
        auto pistonForce = [this](double /*t*/) -> Vec3 {
            Vec3 pinW = crank->bodyToWorld(Vec3(0, R, 0));
            double theta = std::atan2(pinW.z, pinW.y);
            double sinT  = std::sin(theta);
            if (sinT <= 0.0) return Vec3(0, 0, 0); // Only during power stroke
            double envelope = 0.5 * (1.0 + std::cos(theta));
            double F_piston = P_PEAK * A_BORE * envelope;
            return Vec3(0, -F_piston, 0); // World frame force in -Y direction
        };
        sys.addForce(std::make_shared<AppliedForce>(
            piston.get(), pistonForce, Vec3(0, PISTON_JOINT_Y, 0), "PistonForce"));

        sys.initialize();
        history.clear();
        updateXForms();
    }

    void step() {
        for (int i=0;i<subSteps;i++) sys.step(dt);
        static int dbgCnt=0;
        if (++dbgCnt % 300 == 0) {
            // Krank piminin merkezini STEP'ten hesapla → render dünya koord.
            // Krank pimi STEP centroid: (pcx1, pcy1, pcz1)
            double pcx1 = partCenter[1][0], pcy1 = partCenter[1][1], pcz1 = partCenter[1][2];
            // pivot = (0, 0, pcz1), tx = (0,0,0)
            // render = Rc * (centroid - pivot) = Rc * (pcx1, pcy1, 0)
            const double* Rc = xf[1].R;
            double crankPinRender_X = Rc[0]*pcx1 + Rc[1]*pcy1;
            double crankPinRender_Y = Rc[3]*pcx1 + Rc[4]*pcy1;

            // Biyel krank deliği STEP konumu = partCenter[1] ile aynı (pin orada oturuyor)
            // Biyel krank deliği STEP: (pcx1, pcy1, pcz2)
            double pcz2 = partCenter[2][2];
            const double* Rr = xf[2].R;
            double lx2 = pcx1 - xf[2].pivX;
            double ly2 = pcy1 - xf[2].pivY;
            double lz2 = pcz1 - pcz2;       // Z offset farkı
            double rodHoleRender_X = Rr[0]*lx2 + Rr[1]*ly2 + Rr[2]*lz2 + xf[2].tx;
            double rodHoleRender_Y = Rr[3]*lx2 + Rr[4]*ly2 + Rr[5]*lz2 + xf[2].ty;

            printf("[DBG] t=%.3f\n"
                   "  crankPin_render : (%.4f, %.4f)\n"
                   "  rodHole_render  : (%.4f, %.4f)\n"
                   "  diff_render (mm): dX=%.4f  dY=%.4f\n"
                   "  rod_tx=%.4f  rod_ty=%.4f  rod_pivY=%.4f\n",
                sys.getTime(),
                crankPinRender_X, crankPinRender_Y,
                rodHoleRender_X,  rodHoleRender_Y,
                crankPinRender_X - rodHoleRender_X,
                crankPinRender_Y - rodHoleRender_Y,
                xf[2].tx, xf[2].ty, xf[2].pivY);
            fflush(stdout);
        }
        updateXForms();
        history.push_back({sys.getTime(), piston->position.y});
        if ((int)history.size() > 800) history.pop_front();
    }

    static constexpr double STEP_PIN_Y_DBG   = 35.8;   // krank pimi Y (STEP)
    static constexpr double STEP_WRIST_Y_DBG = 132.4;  // bilek pimi Y (STEP)

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

        // Krank milinin STEP dosyasındakı kusursuz fiziksel dönüş ekseni
        // Bounding box Z_axis sol mildeki veya dişlilerdeki keyway/spoke yüzünden kaymasına 
        // karşılık, sağ mil (Part 0) üzerinde test edildiğinde saf eksenin Z=2.69833 olduğu, 
        // Y ekseninin ise istisnasız -2.1680 olduğu kanıtlanmıştır.
        const float TRUE_AXIS_Y   = -2.1680f;    
        const float TRUE_AXIS_Z   = 2.6983f; 
        const float PISTON_Y = 140.6f;

        auto setCrankGroup = [&](int i) {
            setXF(i, 0.0, 0.0, 0.0, Rc);
            xf[i].pivX = 0.0f;
            xf[i].pivY = TRUE_AXIS_Y;
            xf[i].pivZ = TRUE_AXIS_Z; 
        };
        setCrankGroup(0); // crankshaft right (kendi orijinaline döndü, wobble=0)
        setCrankGroup(1); // crank pin 
        setCrankGroup(5); // crankshaft left (artık kendi keyway'inden kurtuldu, wobble=0)
        setCrankGroup(6); // PRIMARY DRIVE GEARS (artık çarpık Z'den kurtuldu, wobble=0)
        setCrankGroup(7); // primary drive gear s 
        setCrankGroup(8); // primary drive gear L 

        // ── Biyel (2) ────────────────────────────────────────────────────────
        float rodPivY = (rodBigEndY + rodSmallEndY) * 0.5f;
        setXF(2, rod->position.x*S, rod->position.y*S, rod->position.z*S, Rr);
        xf[2].pivX = 0.0f;
        xf[2].pivY = rodPivY;
        xf[2].pivZ = TRUE_AXIS_Z; 

        // ── Piston grubu (3,4) ───────────────────────────────────────────────
        setXF(3, piston->position.x*S, piston->position.y*S, piston->position.z*S, Rp);
        xf[3].pivX = 0.0f; xf[3].pivY = PISTON_Y; xf[3].pivZ = TRUE_AXIS_Z;
        setXF(4, piston->position.x*S, piston->position.y*S, piston->position.z*S, Rp);
        xf[4].pivX = 0.0f; xf[4].pivY = PISTON_Y; xf[4].pivZ = TRUE_AXIS_Z;

        // ── Çıkış dişlisi (9,10) ─────────────────────────────────────────────
        double gdX = drivenGear->position.x * S;
        setXF(9,  gdX, drivenGear->position.y*S, drivenGear->position.z*S, Rd);
        xf[9].pivX  = gdX; xf[9].pivY  = TRUE_AXIS_Y; xf[9].pivZ  = TRUE_AXIS_Z;
        setXF(10, gdX, drivenGear->position.y*S, drivenGear->position.z*S, Rd);
        xf[10].pivX = gdX; xf[10].pivY = TRUE_AXIS_Y; xf[10].pivZ = TRUE_AXIS_Z;
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

        // ── Vertex analizi: rod big-end ve crank pin gerçek aks merkezleri ──
        // Part 1 = crank pin: tüm vertex'lerin X/Y/Z ortalama ekseni
        // Part 2 = connecting rod: Y<50mm bölgesi = big end, Y>115mm = small end
        if (loader_.parts.size() >= 3) {
            // Crank pin (part 1): her vertex aynı pim ekseni etrafında
            // Silindirin aksı = X yönünde, merkez = (cx, cy) in YZ
            // → cx = centroid.x, cy = centroid.y (zaten biliyoruz)
            auto& pin = loader_.parts[1];
            float pinX=0,pinY=0,pinZ=0; int pinN=0;
            for (auto& v : pin.verts) { pinX+=v.x; pinY+=v.y; pinZ+=v.z; pinN++; }
            if (pinN) { pinX/=pinN; pinY/=pinN; pinZ/=pinN; }

            // Connecting rod (part 2): big end hole = min-Y bölgesi
            // Delik aksı = rod'un en alt (min-Y) vertex kümesinin Y ortalaması
            // Sadece en alttaki %10 vertex = delik ağzı civarı
            auto& rod = loader_.parts[2];
            // Önce Y min-max bul
            float rodMinY = 1e9f, rodMaxY = -1e9f;
            for (auto& v : rod.verts) { rodMinY=std::min(rodMinY,v.y); rodMaxY=std::max(rodMaxY,v.y); }
            float rodSpan = rodMaxY - rodMinY;
            // Big-end delik = en alt %15 bölgesi (alt flanş + delik çevresi)
            float bigYthresh = rodMinY + rodSpan * 0.15f;
            // Small-end delik = en üst %15 bölgesi
            float smlYthresh = rodMaxY - rodSpan * 0.15f;
            float bigX=0,bigY=0,bigZ=0; int bigN=0;
            float smlX=0,smlY=0,smlZ=0; int smlN=0;
            for (auto& v : rod.verts) {
                if (v.y < bigYthresh) { bigX+=v.x; bigY+=v.y; bigZ+=v.z; bigN++; }
                if (v.y > smlYthresh) { smlX+=v.x; smlY+=v.y; smlZ+=v.z; smlN++; }
            }
            if (bigN) { bigX/=bigN; bigY/=bigN; bigZ/=bigN; }
            if (smlN) { smlX/=smlN; smlY/=smlN; smlZ/=smlN; }
            // Delik MERKEZİ = dış flanştan içeriye doğru bir miktar yukarı:
            // Rod üst flanş bitiş Y + yarı delik yüksekliği ≈ bigYthresh + (pin.cy - rodMinY)
            // Ama en iyi tahmin = crank pin centroid ile aynı Y çünkü onlar mate eder
            // Yani big-end HOLE CENTER Y = crank pin centroid Y = pin.cy
            // Rod büyük uç flanş centroid (bigY) onun altında kalır, Z'si de farklıdır
            float bigHoleCenterY = pin.cy;   // = 35.8mm (crank pin ile eşleşmeli)
            float smlHoleCenterY = loader_.parts[3].cy; // = 132.4mm (wrist pin ile)

            printf("[VERTEX ANALIZ]\n");
            printf("  Crank pin centroid      : (%.4f, %.4f, %.4f) mm\n", pinX, pinY, pinZ);
            printf("  Rod Y range             : min=%.4f max=%.4f span=%.4f mm\n", rodMinY, rodMaxY, rodSpan);
            printf("  Rod big-end  flange ctr : (%.4f, %.4f, %.4f) mm  [N=%d]\n", bigX, bigY, bigZ, bigN);
            printf("  Rod small-end flange ctr: (%.4f, %.4f, %.4f) mm  [N=%d]\n", smlX, smlY, smlZ, smlN);
            printf("  Assumed big hole  Y     : %.4f mm (= crank pin Y)\n", bigHoleCenterY);
            printf("  Assumed small hole Y    : %.4f mm (= wrist pin Y)\n", smlHoleCenterY);
            printf("  Rod pivot Y (midpoint)  : %.4f mm\n", (bigHoleCenterY+smlHoleCenterY)*0.5f);
            
            // CAD HATASI DÜZELTME (GEOMETRİ KAYDIRMA)
            // Motor gövdesinin sağ tarafı Z=2.69833, sol tarafı ve dişliler Z=2.26036'da modellenmiş!
            // Pim ise Z=2.5793 çizilmiş. Bu kaçıklıkları engellemek için hepsini Z=2.69833 eksenine oturtuyoruz
            auto shiftMeshZ = [&](int id, float sz) {
                if (id >= loader_.parts.size()) return;
                for(auto& v : loader_.parts[id].verts) v.z += sz;
                loader_.parts[id].cz += sz;
            };
            
            float shift_pin = 2.69833f - 2.579345f;
            shiftMeshZ(1, shift_pin); // crank pin
            shiftMeshZ(2, shift_pin); // rod
            shiftMeshZ(3, shift_pin); // wrist pin
            shiftMeshZ(4, shift_pin); // piston
            
            float shift_left = 2.69833f - 2.26036f;
            shiftMeshZ(5, shift_left); // crank left
            shiftMeshZ(6, shift_left); // gears
            shiftMeshZ(7, shift_left);
            shiftMeshZ(8, shift_left);
            shiftMeshZ(9, shift_left);
            shiftMeshZ(10, shift_left);

            fflush(stdout);
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
        case Qt::Key_V:
            if (!recording_) startRecording();
            else             stopRecording();
            break;
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
        // Kayıt: mevcut frame'i ffmpeg'e gönder
        if (recording_ && ffmpeg_ && ffmpeg_->state()==QProcess::Running) {
            QImage img(size(), QImage::Format_ARGB32);
            render(&img);
            ffmpeg_->write((const char*)img.constBits(), img.sizeInBytes());
        }
    }

private:
    AssemblyLoader loader_;
    Simulation     sim_;
    Camera         cam_;
    QTimer*        timer_=nullptr;
    QElapsedTimer  elapsed_;
    bool           paused_=false, wireframe_=false;
    bool           recording_=false;
    QProcess*      ffmpeg_=nullptr;
    int            frameCount_=0;
    QPoint         lastMouse_;

    void startRecording() {
        if (recording_) return;
        ffmpeg_ = new QProcess(this);
        QStringList args;
        args << "-y" << "-f" << "rawvideo" << "-pixel_format" << "bgra"
             << "-video_size" << QString("%1x%2").arg(width()).arg(height())
             << "-framerate" << "60"
             << "-i" << "pipe:0"
             << "-c:v" << "libx264" << "-preset" << "fast"
             << "-crf" << "18" << "-pix_fmt" << "yuv420p"
             << "assembly_sim.mp4";
        ffmpeg_->start("ffmpeg", args);
        ffmpeg_->waitForStarted();
        recording_ = true;
        printf("[REC] Kayıt başladı → assembly_sim.mp4 (%dx%d @60fps)\n", width(), height());
        fflush(stdout);
    }
    void stopRecording() {
        if (!recording_) return;
        recording_ = false;
        if (ffmpeg_) {
            ffmpeg_->closeWriteChannel();
            ffmpeg_->waitForFinished(10000);
            printf("[REC] Kayıt tamamlandı → assembly_sim.mp4\n");
            fflush(stdout);
            delete ffmpeg_; ffmpeg_=nullptr;
        }
    }
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
        // Anlık motor torku
        {
            Vec3 pinW = sim_.crank->bodyToWorld(Vec3(0, sim_.R, 0));
            double theta = std::atan2(pinW.z, pinW.y);
            double sinT  = std::sin(theta);
            double envelope = (sinT > 0) ? 0.5*(1.0+std::cos(theta)) : 0.0;
            double F_piston = sim_.P_PEAK * sim_.A_BORE * envelope;
            double tau      = F_piston * sim_.R * std::max(0.0, sinT);
            double P_gas    = sim_.P_ATM + sim_.P_PEAK * envelope;
            p.setPen(QColor(255,180,80));
            L(QString("Gaz P    : %1 bar").arg(P_gas*1e-5,0,'f',1));
            L(QString("Motor τ  : %1 N·m").arg(tau,0,'f',1));
        }
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
        RL("V             Kayıt (MP4)");
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
        if (recording_) {
            p.setPen(QColor(255,40,40));
            p.setFont(QFont("Sans",11,QFont::Bold));
            p.drawText(width()-130, 22, "● REC");
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
