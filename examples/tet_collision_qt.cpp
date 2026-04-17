/**
 * tet_collision_qt.cpp
 * Two ANCF flexible bodies with automatic contact via FlexibleContactManager.
 *
 * Body A : stiff platform, bottom-fixed
 * Body B : soft impactor, free-falling onto A
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
#include <QProcess>
#include <QImage>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <memory>

#include "mb/fem/FlexibleBody.h"
#include "mb/fem/FlexibleIntegrators.h"
#include "mb/fem/FlexibleContactManager.h"

using namespace mb;

// ─── Body parameters ─────────────────────────────────────────────────────────
static constexpr double TET_SIZE = 0.15;          // kenar uzunluğu
static constexpr double EA=1e7, RHO_A=1200.0;    // sert kauçuk
static constexpr double EB=5e6, RHO_B=1000.0;    // orta sert
static constexpr double NU = 0.3;
static constexpr int    NSUB = 200;
static constexpr double RECORD_DURATION = 8.0;  // saniye
static constexpr int    RECORD_FPS = 60;

// Tek tetrahedron mesh oluştur (taban altta)
static GmshMesh makeSingleTet(double sz, double dx, double dy, double dz){
    GmshMesh m;
    double a = sz;
    m.nodes = {
        {1, dx + 0.0,    dy + 0.0,    dz + 0.0},
        {2, dx + a,      dy + 0.0,    dz + 0.0},
        {3, dx + a*0.5,  dy + 0.0,    dz + a*0.866},
        {4, dx + a*0.5,  dy + a*0.816, dz + a*0.289}
    };
    m.elements = {{1, 4, {1,2,3,4}}};
    return m;
}

// Ters tetrahedron: sivri uç altta (y=0), taban yukarıda
static GmshMesh makeInvertedTet(double sz, double dx, double dy, double dz){
    GmshMesh m;
    double a = sz;
    double h = a * 0.816;  // tetrahedron yüksekliği
    // Node 1: sivri uç altta (y=0)
    // Node 2,3,4: üçgen taban yukarıda (y=h)
    m.nodes = {
        {1, dx + a*0.5,  dy + 0.0,    dz + a*0.289},  // tepe (altta)
        {2, dx + 0.0,    dy + h,      dz + 0.0},       // taban köşe 1
        {3, dx + a,      dy + h,      dz + 0.0},       // taban köşe 2
        {4, dx + a*0.5,  dy + h,      dz + a*0.866}    // taban köşe 3
    };
    m.elements = {{1, 4, {1,2,3,4}}};
    return m;
}

// ─── Mesh helper ─────────────────────────────────────────────────────────────
static void shiftMesh(GmshMesh& m, double dx, double dy, double dz){
    for(auto& n:m.nodes){n.x+=dx;n.y+=dy;n.z+=dz;}
}

static Vec3 bodyCentroid(const std::vector<Vec3>& pos){
    Vec3 c{}; for(auto& p:pos){c.x+=p.x;c.y+=p.y;c.z+=p.z;}
    double n=pos.size(); return{c.x/n,c.y/n,c.z/n};
}

// ─── Colour map ──────────────────────────────────────────────────────────────
// ─── Simulation ──────────────────────────────────────────────────────────────
struct Sim {
    std::shared_ptr<FlexibleBody>           bodyA, bodyB;
    std::unique_ptr<ImplicitFlexIntegrator> intA,  intB;
    std::vector<std::array<int,4>>          tetA,  tetB;

    std::unique_ptr<FlexibleContactManager> contactMgr;

    std::vector<Vec3>   posA, posB;

    double time=0; bool gravityOn=true;

    void build(){
        // Contact config
        FlexContactConfig cfg;
        cfg.hertzExponent = 1.5;
        cfg.maxStiffness = 1e6;
        contactMgr = std::make_unique<FlexibleContactManager>(cfg);

        // Body A: ters tetrahedron, sivri ucu ground'da
        {
            auto mesh=makeInvertedTet(TET_SIZE, 0.0, 0.0, 0.0);
            ElasticMaterialProps mat{EA,NU,RHO_A,MaterialType::NeoHookean};
            bodyA=FlexibleBody::fromMesh(mesh,mat,"A",true);
            bodyA->gravity={0,-9.81,0}; bodyA->dampingAlpha=5.0;
            bodyA->fixNodesOnPlane('y',0.0,1e-6);  // sivri ucu sabitler
            intA=std::make_unique<ImplicitFlexIntegrator>(*bodyA);
            intA->hhtAlpha=-0.1; intA->newtonTol=1e-4; intA->maxNewtonIter=30;
        }
        // Body B: üst tetrahedron, serbest düşüş
        {
            double hA = TET_SIZE * 0.816;  // A'nın yüksekliği
            auto mesh=makeSingleTet(TET_SIZE, 0.0, hA+0.12, 0.0);
            ElasticMaterialProps mat{EB,NU,RHO_B,MaterialType::NeoHookean};
            bodyB=FlexibleBody::fromMesh(mesh,mat,"B",true);
            bodyB->gravity={0,-9.81,0}; bodyB->dampingAlpha=5.0;
            intB=std::make_unique<ImplicitFlexIntegrator>(*bodyB);
            intB->hhtAlpha=-0.1; intB->newtonTol=1e-4; intB->maxNewtonIter=30;
        }
        tetA=bodyA->getTetConnectivity();
        tetB=bodyB->getTetConnectivity();
        // Register bodies with contact manager
        FlexContactMaterial matA{0.8, 0.3, EA, NU};
        FlexContactMaterial matB{0.8, 0.3, EB, NU};
        contactMgr->addBody(*bodyA, matA);
        contactMgr->addBody(*bodyB, matB);
        contactMgr->enableGround(GroundPlane{0.0, {0,1,0}});
        contactMgr->setContactMargin(0.002);
        contactMgr->setMaxDepth(0.005);
        contactMgr->maxForcePerDof = 800.0;

        time=0;
        updateRenderData();
    }

    void step(double dt){
        double ds=dt/NSUB;
        Vec3 g=gravityOn?Vec3{0,-9.81,0}:Vec3{0,0,0};
        bodyA->gravity=g; bodyB->gravity=g;

        for(int s=0;s<NSUB;s++){
            contactMgr->step();

            // Apply contact forces via externalForces callback
            auto fA = contactMgr->getContactForces(bodyA->id);
            auto fB = contactMgr->getContactForces(bodyB->id);
            if(!fA.empty())
                bodyA->externalForces=[fA](FlexibleBody&){return fA;};
            else
                bodyA->externalForces=nullptr;
            if(!fB.empty())
                bodyB->externalForces=[fB](FlexibleBody&){return fB;};
            else
                bodyB->externalForces=nullptr;

            intA->step(ds);
            intB->step(ds);

            // Velocity clamp
            auto clampVel=[](FlexibleBody& b, double vmax){
                auto qd=b.getFlexQd(); bool clamped=false;
                int nn=b.nodes.size();
                for(int i=0;i<nn;i++){
                    double vx=qd[i*12],vy=qd[i*12+1],vz=qd[i*12+2];
                    double vm=std::sqrt(vx*vx+vy*vy+vz*vz);
                    if(vm>vmax){
                        double sc=vmax/vm;
                        for(int d=0;d<12;d++) qd[i*12+d]*=sc;
                        clamped=true;
                    }
                }
                if(clamped) b.setFlexQd(qd);
            };
            clampVel(*bodyA,10.0);
            clampVel(*bodyB,10.0);
        }
        time+=dt;
        updateRenderData();
    }

    void updateRenderData(){
        posA=bodyA->getNodePositions();
        posB=bodyB->getNodePositions();
    }

    int numContacts() const { return contactMgr ? contactMgr->numContacts() : 0; }

    double kineticEnergy() const {
        double ke=0;
        for(auto* b:{bodyA.get(),bodyB.get()}){
            auto qd=b->getFlexQd(); auto md=b->getMassDiagonal();
            for(int i=0;i<(int)qd.size();i++) ke+=0.5*md[i]*qd[i]*qd[i];
        }
        return ke;
    }
};

// ─── Qt Widget ───────────────────────────────────────────────────────────────
class TetCollisionWidget : public QWidget {
public:
    TetCollisionWidget(QWidget* parent=nullptr):QWidget(parent){
        setWindowTitle("MBC++ — İki Tetrahedron Çarpışması");
        resize(1200,700);
        sim_.build();
        timer_=new QTimer(this);
        connect(timer_,&QTimer::timeout,this,&TetCollisionWidget::tick);
        timer_->start(16);
        elapsed_.start();
        setFocusPolicy(Qt::StrongFocus);
        azimuth_=0.5; elevation_=0.28;

        // Start ffmpeg recording
        startRecording();
    }

    ~TetCollisionWidget(){
        stopRecording();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this); p.setRenderHint(QPainter::Antialiasing);
        int w=width(),h=height();
        p.fillRect(rect(),QColor(15,17,22));

        double scale=zoom_*std::min(w,h)*0.35;
        double cx=w*0.5+panX_, cy=h*0.55+panY_;
        double ca=std::cos(azimuth_),sa=std::sin(azimuth_);
        double cEl=std::cos(elevation_),sEl=std::sin(elevation_);
        auto toScreen=[&](Vec3 v)->QPointF{
            double x1=v.x*ca+v.z*sa,y1=v.y,z1=-v.x*sa+v.z*ca;
            return{cx+x1*scale,cy-(y1*cEl-z1*sEl)*scale};
        };

        // Ground grid
        p.setPen(QPen(QColor(30,35,48),1));
        for(double m=-0.5;m<=1.0+1e-6;m+=0.1)
            p.drawLine(toScreen(Vec3{m,0,-0.3}),toScreen(Vec3{m,0,0.4}));
        for(double m=-0.3;m<=0.4+1e-6;m+=0.1)
            p.drawLine(toScreen(Vec3{-0.5,0,m}),toScreen(Vec3{1.0,0,m}));

        // Coord axes
        auto ax=[&](Vec3 tip,QColor c,const QString& l){
            p.setPen(QPen(c,2));
            p.drawLine(toScreen(Vec3{0,0,0}),toScreen(tip));
            p.setFont(QFont("Sans",9,QFont::Bold));
            p.drawText(toScreen(tip)+QPointF(4,-4),l);
        };
        ax(Vec3{0.12,0,0},QColor(220,70,70),"X");
        ax(Vec3{0,0.12,0},QColor(70,200,70),"Y");
        ax(Vec3{0,0,0.12},QColor(70,130,255),"Z");

        // Draw a body
        auto drawBody=[&](const std::vector<Vec3>& pos,
                          const std::vector<std::array<int,4>>& tets,
                          QColor tint)
        {
            static const int fi[4][3]={{1,2,3},{0,2,3},{0,1,3},{0,1,2}};
            static const int ei[6][2]={{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
            QColor fc(tint.red(),tint.green(),tint.blue(),165);
            for(int e=0;e<(int)tets.size();e++){
                QPointF pts[4];
                for(int k=0;k<4;k++) pts[k]=toScreen(pos[tets[e][k]]);
                std::array<std::pair<double,int>,4> fd;
                for(int f=0;f<4;f++){
                    double d=0;
                    for(int k=0;k<3;k++){auto& vk=pos[tets[e][fi[f][k]]];d+=(-vk.x*sa+vk.z*ca);}
                    fd[f]={d/3,f};
                }
                std::sort(fd.begin(),fd.end());
                for(auto&[depth,f]:fd){
                    QPainterPath path;
                    path.moveTo(pts[fi[f][0]]); path.lineTo(pts[fi[f][1]]);
                    path.lineTo(pts[fi[f][2]]); path.closeSubpath();
                    p.fillPath(path,fc);
                }
                // Kenar çizgileri (kalın beyaz)
                p.setPen(QPen(QColor(255,255,255),2.5));
                for(int k=0;k<6;k++)
                    p.drawLine(pts[ei[k][0]],pts[ei[k][1]]);
            }
        };

        Vec3 cA=bodyCentroid(sim_.posA),cB=bodyCentroid(sim_.posB);
        if((-cA.x*sa+cA.z*ca)<(-cB.x*sa+cB.z*ca)){
            drawBody(sim_.posA,sim_.tetA,QColor(200,110,40));
            drawBody(sim_.posB,sim_.tetB,QColor(40,110,220));
        } else {
            drawBody(sim_.posB,sim_.tetB,QColor(40,110,220));
            drawBody(sim_.posA,sim_.tetA,QColor(200,110,40));
        }

        drawHUD(p,w,h);
    }

    void keyPressEvent(QKeyEvent* e) override {
        switch(e->key()){
        case Qt::Key_Space: paused_=!paused_; break;
        case Qt::Key_R: sim_.build(); elapsed_.restart(); frameCount_=0; break;
        case Qt::Key_G: sim_.gravityOn=!sim_.gravityOn; break;
        case Qt::Key_E:{
            auto qd=sim_.bodyB->getFlexQd();
            int nn=sim_.bodyB->nodes.size();
            for(int i=0;i<nn;i++){qd[i*12+0]+=3.0;qd[i*12+2]+=1.5;}
            sim_.bodyB->setFlexQd(qd); break;}
        case Qt::Key_Plus:  case Qt::Key_Equal: zoom_*=1.2; break;
        case Qt::Key_Minus: zoom_/=1.2; break;
        case Qt::Key_Escape: close(); break;
        }
        update();
    }
    void mousePressEvent(QMouseEvent* e) override {lastMousePos_=e->pos();e->accept();}
    void mouseMoveEvent(QMouseEvent* e) override {
        QPoint d=e->pos()-lastMousePos_; lastMousePos_=e->pos();
        if(e->buttons()&Qt::LeftButton){
            azimuth_-=d.x()*0.005; elevation_-=d.y()*0.005;
            elevation_=std::clamp(elevation_,-1.5,1.5);
        }
        if(e->buttons()&(Qt::RightButton|Qt::MiddleButton)){panX_+=d.x();panY_+=d.y();}
        update(); e->accept();
    }
    void wheelEvent(QWheelEvent* e) override {
        zoom_*=(e->angleDelta().y()>0)?1.15:1.0/1.15; update(); e->accept();
    }

private slots:
    void tick(){
        double dt=1.0/RECORD_FPS;
        if(!paused_) sim_.step(dt);
        frameCount_++;
        repaint();  // synchronous paint
        // Write frame to ffmpeg after paint
        if(recording_ && ffmpeg_ && ffmpeg_->state()==QProcess::Running){
            QImage img=grab().toImage().convertToFormat(QImage::Format_RGBA8888);
            ffmpeg_->write((const char*)img.constBits(), img.sizeInBytes());
            recordedFrames_++;
            if(recordedFrames_ >= RECORD_FPS * RECORD_DURATION){
                stopRecording();
            }
        }
    }

private:
    Sim sim_;
    QTimer* timer_=nullptr; QElapsedTimer elapsed_;
    bool paused_=false; int frameCount_=0;
    double zoom_=1.0,azimuth_=0,elevation_=0,panX_=0,panY_=0;
    QPoint lastMousePos_;

    // Recording
    QProcess* ffmpeg_=nullptr;
    bool recording_=false;
    int recordedFrames_=0;

    void startRecording(){
        ffmpeg_=new QProcess(this);
        QString cmd="ffmpeg";
        QStringList args;
        args<<"-y"
            <<"-f"<<"rawvideo"
            <<"-pixel_format"<<"rgba"
            <<"-video_size"<<QString("%1x%2").arg(width()).arg(height())
            <<"-framerate"<<QString::number(RECORD_FPS)
            <<"-i"<<"pipe:0"
            <<"-c:v"<<"libx264"
            <<"-preset"<<"fast"
            <<"-crf"<<"18"
            <<"-pix_fmt"<<"yuv420p"
            <<"-movflags"<<"+faststart"
            <<"tet_collision.mp4";
        ffmpeg_->setWorkingDirectory(QApplication::applicationDirPath());
        ffmpeg_->start(cmd,args);
        ffmpeg_->waitForStarted(3000);
        recording_=true;
        recordedFrames_=0;
        qDebug("Recording started: %dx%d @ %d fps, %.1f s",width(),height(),RECORD_FPS,RECORD_DURATION);
    }

    void stopRecording(){
        if(!recording_) return;
        recording_=false;
        if(ffmpeg_){
            ffmpeg_->closeWriteChannel();
            ffmpeg_->waitForFinished(10000);
            qDebug("Recording finished: %d frames -> tet_collision.mp4",recordedFrames_);
            ffmpeg_->deleteLater(); ffmpeg_=nullptr;
        }
    }

    void drawHUD(QPainter& p,int w,int){
        double fps=frameCount_/(elapsed_.elapsed()*0.001+1e-9);
        int nc=sim_.numContacts();
        p.setPen(QColor(200,200,200));
        QFont f("Monospace",11); f.setStyleHint(QFont::Monospace); p.setFont(f);
        int x=14,y=24,dy=20;
        auto line=[&](const QString& s){p.drawText(x,y,s);y+=dy;};

        line(QString("t = %1 s").arg(sim_.time,0,'f',3));
        line(QString("FPS  %1").arg(fps,0,'f',1));
        line(QString("Substeps/frame: %1").arg(NSUB));
        line("");

        if(nc>0){
            p.setPen(QColor(255,80,80));
            QFont fb("Monospace",12,QFont::Bold); p.setFont(fb);
            p.drawText(x,y,QString("● CONTACT  (%1 pts)").arg(nc));
            y+=dy+2; p.setFont(f); p.setPen(QColor(200,200,200));
        } else {
            p.setPen(QColor(80,200,80)); line("○ no contact"); p.setPen(QColor(200,200,200));
        }
        line("");
        line(QString("A: E=%1 Pa  ρ=%2").arg(EA,0,'e',1).arg(int(RHO_A)));
        line(QString("B: E=%1 Pa  ρ=%2").arg(EB,0,'e',1).arg(int(RHO_B)));
        line("");
        line(QString("DOF A: %1  B: %2").arg(sim_.bodyA->numDof).arg(sim_.bodyB->numDof));
        line(QString("KE: %1 J").arg(sim_.kineticEnergy(),0,'e',3));
        line(QString("Gravity: %1").arg(sim_.gravityOn?"ON":"OFF"));

        p.setPen(QColor(120,120,140));
        int bx=w-260,by2=24;
        auto r=[&](const QString& s){p.drawText(bx,by2,s);by2+=18;};
        r("Sol sürükle    Döndür");
        r("Sağ sürükle    Kaydır");
        r("Tekerlek       Zoom");
        r("E              İtki");
        r("G              Gravity");
        r("SPACE          Duraklat");
        r("R              Sıfırla");
        r("ESC            Çıkış");
        p.setPen(QColor(255,150,80));
        p.drawText(bx,by2+10,"■ A: alt tetra (turuncu)");
        p.setPen(QColor(80,150,255));
        p.drawText(bx,by2+28,"■ B: üst tetra (mavi)");

        if(paused_){
            p.setPen(QColor(255,80,80));
            QFont fb("Sans",16,QFont::Bold); p.setFont(fb);
            p.drawText(w/2-50,40,"⏸  DURAKLATILDI");
        }
    }
};

int main(int argc,char* argv[]){
    QApplication app(argc,argv);
    TetCollisionWidget win; win.show();
    return app.exec();
}
