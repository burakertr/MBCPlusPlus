#include "mb/fem/FlexibleBody.h"
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <map>

namespace mb {

int FlexibleBody::flexNextId_ = 0;

FlexibleBody::FlexibleBody(const std::string& name)
    : Body(BodyType::DYNAMIC, name.empty() ? "FlexBody_" + std::to_string(flexNextId_++) : name)
{
}

// ─── Construction ────────────────────────────────────────────

std::shared_ptr<FlexibleBody> FlexibleBody::fromMesh(
    const GmshMesh& mesh,
    const ElasticMaterialProps& matProps,
    const std::string& name,
    bool highOrderQuad)
{
    auto body = std::make_shared<FlexibleBody>(name);
    body->materialProps = matProps;
    body->material = createMaterial(matProps);

    // Build node-id → array-index map
    std::map<int, int> nodeIdMap;
    for (int i = 0; i < (int)mesh.nodes.size(); i++) {
        const auto& gn = mesh.nodes[i];
        nodeIdMap[gn.id] = i;

        ANCFNode nd;
        nd.id = i;
        nd.X0 = {gn.x, gn.y, gn.z};
        nd.q.fill(0);
        nd.qd.fill(0);
        nd.qdd.fill(0);
        nd.fixed = false;
        nd.fixedDOF.fill(false);

        // Position = reference position
        nd.q[0] = gn.x; nd.q[1] = gn.y; nd.q[2] = gn.z;
        // Gradient = identity (undeformed)
        nd.q[3] = 1;  nd.q[7] = 1;  nd.q[11] = 1;

        body->nodes.push_back(nd);
    }

    // Build elements
    for (const auto& ge : mesh.elements) {
        TetConnectivity conn;
        for (int k = 0; k < 4; k++) {
            auto it = nodeIdMap.find(ge.nodeIds[k]);
            if (it == nodeIdMap.end())
                throw std::runtime_error("Node " + std::to_string(ge.nodeIds[k]) + " not found");
            conn.nodeIds[k] = it->second;
        }
        body->elements.emplace_back(conn, body->nodes, *body->material, highOrderQuad);
    }

    // Lock gradient DOFs (3..11 per node).  The current formulation computes
    // F = Σ dNa/dX · r_a  (standard linear-tet), so elastic forces only act
    // on position DOFs.  Gradient DOFs carry mass but no stiffness — leaving
    // them free causes immediate divergence.  Fixing them gives a correct
    // standard FEM with 3 translational DOFs per node.
    for (auto& nd : body->nodes) {
        for (int d = 3; d < ANCF_NODE_DOF; d++)
            nd.fixedDOF[d] = true;
    }

    body->numDof = (int)body->nodes.size() * ANCF_NODE_DOF;
    body->rebuildDofMap();

    return body;
}

// ─── Boundary Conditions ─────────────────────────────────────

void FlexibleBody::fixNode(int nodeIdx) {
    nodes[nodeIdx].fixed = true;
    nodes[nodeIdx].fixedDOF.fill(true);
    rebuildDofMap();
}

void FlexibleBody::fixNodeDOFs(int nodeIdx, const std::vector<int>& dofs) {
    for (int d : dofs)
        nodes[nodeIdx].fixedDOF[d] = true;
    rebuildDofMap();
}

void FlexibleBody::fixNodesOnPlane(char axis, double value, double tol) {
    int ax = (axis == 'x') ? 0 : (axis == 'y') ? 1 : 2;
    for (int i = 0; i < (int)nodes.size(); i++) {
        if (std::abs(nodes[i].X0[ax] - value) < tol)
            fixNode(i);
    }
}

void FlexibleBody::rebuildDofMap() {
    fixedDofMask_.clear();
    freeDofMap_.clear();
    for (int i = 0; i < (int)nodes.size(); i++) {
        for (int d = 0; d < ANCF_NODE_DOF; d++) {
            bool isFixed = nodes[i].fixedDOF[d];
            fixedDofMask_.push_back(isFixed);
            if (!isFixed)
                freeDofMap_.push_back(i * ANCF_NODE_DOF + d);
        }
    }
}

bool FlexibleBody::isDofFixed(int globalDof) const {
    if (globalDof < 0 || globalDof >= (int)fixedDofMask_.size()) return false;
    return fixedDofMask_[globalDof];
}

// ─── State Management ────────────────────────────────────────

std::vector<double> FlexibleBody::getFlexQ() const {
    std::vector<double> q(numDof);
    for (int i = 0; i < (int)nodes.size(); i++)
        for (int d = 0; d < ANCF_NODE_DOF; d++)
            q[i * ANCF_NODE_DOF + d] = nodes[i].q[d];
    return q;
}

void FlexibleBody::setFlexQ(const std::vector<double>& q) {
    for (int i = 0; i < (int)nodes.size(); i++) {
        int off = i * ANCF_NODE_DOF;
        for (int d = 0; d < ANCF_NODE_DOF; d++)
            if (!nodes[i].fixedDOF[d])
                nodes[i].q[d] = q[off + d];
    }
}

std::vector<double> FlexibleBody::getFlexQd() const {
    std::vector<double> qd(numDof);
    for (int i = 0; i < (int)nodes.size(); i++)
        for (int d = 0; d < ANCF_NODE_DOF; d++)
            qd[i * ANCF_NODE_DOF + d] = nodes[i].qd[d];
    return qd;
}

void FlexibleBody::setFlexQd(const std::vector<double>& qd) {
    for (int i = 0; i < (int)nodes.size(); i++) {
        int off = i * ANCF_NODE_DOF;
        for (int d = 0; d < ANCF_NODE_DOF; d++)
            if (!nodes[i].fixedDOF[d])
                nodes[i].qd[d] = qd[off + d];
    }
}

void FlexibleBody::setAngularVelocityFlex(const Vec3& omega) {
    angularVelocity = omega;
    int n = (int)nodes.size();
    double cx = 0, cy = 0, cz = 0;
    for (const auto& nd : nodes) { cx += nd.q[0]; cy += nd.q[1]; cz += nd.q[2]; }
    cx /= n; cy /= n; cz /= n;

    for (auto& nd : nodes) {
        double rx = nd.q[0]-cx, ry = nd.q[1]-cy, rz = nd.q[2]-cz;
        nd.qd[0] = omega.y*rz - omega.z*ry;
        nd.qd[1] = omega.z*rx - omega.x*rz;
        nd.qd[2] = omega.x*ry - omega.y*rx;

        // Ḟ = Ω·F (column-major gradient DOFs)
        for (int col = 0; col < 3; col++) {
            int off = 3 + col*3;
            double f0 = nd.q[off], f1 = nd.q[off+1], f2 = nd.q[off+2];
            nd.qd[off]   = -omega.z*f1 + omega.y*f2;
            nd.qd[off+1] =  omega.z*f0 - omega.x*f2;
            nd.qd[off+2] = -omega.y*f0 + omega.x*f1;
        }
    }
}

void FlexibleBody::setLinearVelocityFlex(const Vec3& v) {
    velocity = v;
    for (auto& nd : nodes) {
        nd.qd[0] = v.x; nd.qd[1] = v.y; nd.qd[2] = v.z;
    }
}

// ─── Mass Matrix ─────────────────────────────────────────────

const std::vector<double>& FlexibleBody::getGlobalMassMatrix() {
    if (!globalMass_.empty()) return globalMass_;

    int n = numDof;
    globalMass_.resize(n * n, 0.0);

    for (const auto& elem : elements) {
        if (!elem.alive) continue;
        auto Me = elem.computeMassMatrix(nodes);
        const auto& nids = elem.nodeIds;

        for (int a = 0; a < 4; a++)
            for (int b = 0; b < 4; b++) {
                int gA = nids[a] * ANCF_NODE_DOF;
                int gB = nids[b] * ANCF_NODE_DOF;
                for (int i = 0; i < ANCF_NODE_DOF; i++)
                    for (int j = 0; j < ANCF_NODE_DOF; j++) {
                        int eRow = a * ANCF_NODE_DOF + i;
                        int eCol = b * ANCF_NODE_DOF + j;
                        globalMass_[(gA+i)*n + (gB+j)] += Me[eRow*48+eCol];
                    }
            }
    }

    // Zero rows/cols for fixed DOFs
    for (int d = 0; d < n; d++) {
        if (fixedDofMask_[d]) {
            for (int j = 0; j < n; j++) {
                globalMass_[d*n+j] = 0;
                globalMass_[j*n+d] = 0;
            }
            globalMass_[d*n+d] = 1;
        }
    }

    return globalMass_;
}

std::vector<double> FlexibleBody::getMassDiagonal() {
    if (!massDiag_.empty()) return massDiag_;
    const auto& M = getGlobalMassMatrix();
    int n = numDof;
    massDiag_.resize(n);
    for (int i = 0; i < n; i++)
        massDiag_[i] = M[i*n+i];
    return massDiag_;
}

std::vector<double> FlexibleBody::getMassDiagonalInverse() {
    if (!massDiagInv_.empty()) return massDiagInv_;
    auto diag = getMassDiagonal();
    int n = numDof;
    massDiagInv_.resize(n);

    double maxTransMass = 0;
    for (int i = 0; i < n; i++) {
        int localDof = i % ANCF_NODE_DOF;
        if (localDof < 3 && diag[i] > maxTransMass)
            maxTransMass = diag[i];
    }
    double minMass = std::max(maxTransMass * 0.1, 1e-12);

    for (int i = 0; i < n; i++)
        massDiagInv_[i] = diag[i] > minMass ? 1.0/diag[i] : 1.0/minMass;

    return massDiagInv_;
}

// ─── Force Computation ───────────────────────────────────────

std::vector<double> FlexibleBody::computeElasticForces() {
    int n = numDof;
    std::vector<double> Q(n, 0.0);
    int nElem = (int)elements.size();

    #pragma omp parallel
    {
        std::vector<double> Qloc(n, 0.0);
        #pragma omp for schedule(dynamic) nowait
        for (int ei = 0; ei < nElem; ei++) {
            if (!elements[ei].alive) continue;
            const auto& elem = elements[ei];
            auto Qe = elem.computeElasticForces(nodes);
            const auto& nids = elem.nodeIds;
            for (int a = 0; a < 4; a++) {
                int gOff = nids[a] * ANCF_NODE_DOF;
                for (int d = 0; d < ANCF_NODE_DOF; d++)
                    Qloc[gOff+d] += Qe[a*ANCF_NODE_DOF+d];
            }
        }
        #pragma omp critical
        for (int i = 0; i < n; i++) Q[i] += Qloc[i];
    }

    for (int d = 0; d < n; d++)
        if (fixedDofMask_[d]) Q[d] = 0;

    return Q;
}

std::vector<double> FlexibleBody::computeGravityForces() {
    int n = numDof;
    std::vector<double> Q(n, 0.0);
    const auto& Mdiag = getMassDiagonal();
    double gVec[3] = {gravity.x, gravity.y, gravity.z};

    for (int i = 0; i < (int)nodes.size(); i++) {
        int off = i * ANCF_NODE_DOF;
        // Only position DOFs get gravity
        for (int d = 0; d < 3; d++) {
            if (!fixedDofMask_[off+d])
                Q[off+d] = Mdiag[off+d] * gVec[d];
        }
    }

    return Q;
}

std::vector<double> FlexibleBody::computeTotalForces() {
    auto Qe = computeElasticForces();
    auto Qg = computeGravityForces();
    int n = numDof;
    std::vector<double> Q(n);

    for (int i = 0; i < n; i++)
        Q[i] = Qe[i] + Qg[i];

    // External forces
    if (externalForces) {
        auto Qext = externalForces(*this);
        for (int i = 0; i < n; i++) Q[i] += Qext[i];
    }

    // Rayleigh damping (mass-proportional, deformation velocity only)
    if (dampingAlpha > 0) {
        auto Mdiag = getMassDiagonal();
        int nNodes = (int)nodes.size();

        // Compute centroid velocity
        double vx = 0, vy = 0, vz = 0;
        for (const auto& nd : nodes) {
            vx += nd.qd[0]; vy += nd.qd[1]; vz += nd.qd[2];
        }
        vx /= nNodes; vy /= nNodes; vz /= nNodes;

        int nNodes2 = (int)nodes.size();
        #pragma omp parallel for schedule(static)
        for (int ni2 = 0; ni2 < nNodes2; ni2++) {
            const auto& nd = nodes[ni2];
            int off = nd.id * ANCF_NODE_DOF;
            double defVx = nd.qd[0] - vx;
            double defVy = nd.qd[1] - vy;
            double defVz = nd.qd[2] - vz;

            if (!fixedDofMask_[off])   Q[off]   -= dampingAlpha * Mdiag[off]   * defVx;
            if (!fixedDofMask_[off+1]) Q[off+1] -= dampingAlpha * Mdiag[off+1] * defVy;
            if (!fixedDofMask_[off+2]) Q[off+2] -= dampingAlpha * Mdiag[off+2] * defVz;

            for (int d = 3; d < ANCF_NODE_DOF; d++) {
                int dofIdx = off + d;
                if (!fixedDofMask_[dofIdx])
                    Q[dofIdx] -= dampingAlpha * Mdiag[dofIdx] * nd.qd[d];
            }
        }
    }

    // Gradient penalty: pull F toward nearest rotation (using R≈I approximation)
    if (gradientPenalty > 0) {
        int nNodesGP = (int)nodes.size();
        #pragma omp parallel for schedule(static)
        for (int ni = 0; ni < nNodesGP; ni++) {
            int off = ni * ANCF_NODE_DOF;
            const auto& nd = nodes[ni];

            // Extract F, compute nearest rotation via iterative polar decomposition
            double f00 = nd.q[3], f10 = nd.q[4], f20 = nd.q[5];
            double f01 = nd.q[6], f11 = nd.q[7], f21 = nd.q[8];
            double f02 = nd.q[9], f12 = nd.q[10], f22 = nd.q[11];

            for (int iter = 0; iter < 3; iter++) {
                double det = f00*(f11*f22-f12*f21) - f01*(f10*f22-f12*f20) + f02*(f10*f21-f11*f20);
                if (std::abs(det) < 1e-20) break;
                double invDet = 1.0/det;
                double c00=(f11*f22-f12*f21)*invDet, c10=-(f01*f22-f02*f21)*invDet, c20=(f01*f12-f02*f11)*invDet;
                double c01=-(f10*f22-f12*f20)*invDet, c11=(f00*f22-f02*f20)*invDet, c21=-(f00*f12-f02*f10)*invDet;
                double c02=(f10*f21-f11*f20)*invDet, c12=-(f00*f21-f01*f20)*invDet, c22=(f00*f11-f01*f10)*invDet;
                f00=0.5*(f00+c00); f10=0.5*(f10+c10); f20=0.5*(f20+c20);
                f01=0.5*(f01+c01); f11=0.5*(f11+c11); f21=0.5*(f21+c21);
                f02=0.5*(f02+c02); f12=0.5*(f12+c12); f22=0.5*(f22+c22);
            }

            double R[9] = {f00,f10,f20, f01,f11,f21, f02,f12,f22};
            for (int d = 0; d < 9; d++) {
                int gDof = off + 3 + d;
                if (!fixedDofMask_[gDof])
                    Q[gDof] -= gradientPenalty * (nd.q[3+d] - R[d]);
            }
        }
    }

    // Zero fixed DOFs
    for (int i = 0; i < n; i++)
        if (fixedDofMask_[i]) Q[i] = 0;

    return Q;
}

// ─── Analysis ────────────────────────────────────────────────

double FlexibleBody::computeStrainEnergy() const {
    double W = 0;
    int nElem = (int)elements.size();
    #pragma omp parallel for reduction(+:W) schedule(dynamic)
    for (int ei = 0; ei < nElem; ei++) {
        if (!elements[ei].alive) continue;
        double F[9];
        elements[ei].computeDeformationGradient(0.25, 0.25, 0.25, nodes, F);
        W += material->strainEnergyDensity(F) * elements[ei].V0;
    }
    return W;
}

double FlexibleBody::computeElementVonMises(int elemIdx) const {
    if (elemIdx < 0 || elemIdx >= (int)elements.size()) return 0;
    if (!elements[elemIdx].alive) return 0;

    // Deformation gradient at centroid (ξ=η=ζ=0.25 for tet)
    double F[9];
    elements[elemIdx].computeDeformationGradient(0.25, 0.25, 0.25, nodes, F);

    // Second Piola-Kirchhoff stress S
    double S[9];
    material->secondPiolaStress(F, S);

    // Cauchy stress: σ = (1/J) F · S · Fᵀ
    double J = mat3util::det(F);
    if (std::abs(J) < 1e-20) J = 1e-20;
    double invJ = 1.0 / std::abs(J);

    // T = F · S
    double T[9];
    mat3util::mul(F, S, T);

    // σ = invJ * T · Fᵀ
    double sigma[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            sigma[i*3+j] = 0;
            for (int k = 0; k < 3; k++)
                sigma[i*3+j] += T[i*3+k] * F[j*3+k];  // Fᵀ_kj = F_jk
            sigma[i*3+j] *= invJ;
        }

    // von Mises: σ_vM = √(½[(σ11-σ22)²+(σ22-σ33)²+(σ33-σ11)²+6(σ12²+σ23²+σ13²)])
    double s11 = sigma[0], s22 = sigma[4], s33 = sigma[8];
    double s12 = sigma[1], s23 = sigma[5], s13 = sigma[2];
    double vm2 = 0.5 * ((s11-s22)*(s11-s22) + (s22-s33)*(s22-s33) + (s33-s11)*(s33-s11)
                       + 6.0*(s12*s12 + s23*s23 + s13*s13));
    return std::sqrt(std::max(vm2, 0.0));
}

// ─── Element Removal (Fracture) ──────────────────────────────

void FlexibleBody::removeElements(const std::vector<int>& elemIndices) {
    if (elemIndices.empty()) return;

    for (int idx : elemIndices) {
        if (idx >= 0 && idx < (int)elements.size())
            elements[idx].alive = false;
    }

    invalidateMassCache();
}

void FlexibleBody::invalidateMassCache() {
    globalMass_.clear();
    massDiag_.clear();
    massDiagInv_.clear();
}

double FlexibleBody::getTotalMass() const {
    double mass = 0;
    for (const auto& elem : elements)
        if (elem.alive) mass += material->density() * elem.V0;
    return mass;
}

double FlexibleBody::getMaxDisplacement() const {
    double maxD = 0;
    for (const auto& nd : nodes) {
        double dx = nd.q[0]-nd.X0[0], dy = nd.q[1]-nd.X0[1], dz = nd.q[2]-nd.X0[2];
        double d = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (d > maxD) maxD = d;
    }
    return maxD;
}

std::vector<Vec3> FlexibleBody::getNodePositions() const {
    std::vector<Vec3> pos;
    pos.reserve(nodes.size());
    for (const auto& nd : nodes)
        pos.emplace_back(nd.q[0], nd.q[1], nd.q[2]);
    return pos;
}

std::vector<Vec3> FlexibleBody::getNodeDisplacements() const {
    std::vector<Vec3> disp;
    disp.reserve(nodes.size());
    for (const auto& nd : nodes)
        disp.emplace_back(nd.q[0]-nd.X0[0], nd.q[1]-nd.X0[1], nd.q[2]-nd.X0[2]);
    return disp;
}

std::vector<std::array<int,4>> FlexibleBody::getTetConnectivity() const {
    std::vector<std::array<int,4>> conn;
    conn.reserve(elements.size());
    for (const auto& e : elements)
        if (e.alive) conn.push_back(e.nodeIds);
    return conn;
}

// ─── Body Interface ──────────────────────────────────────────

std::vector<double> FlexibleBody::getQ() const { return getFlexQ(); }
void FlexibleBody::setQ(const std::vector<double>& q) { setFlexQ(q); }
std::vector<double> FlexibleBody::getV() const { return getFlexQd(); }
void FlexibleBody::setV(const std::vector<double>& v) { setFlexQd(v); }

std::vector<double> FlexibleBody::computeQDot() const { return getFlexQd(); }

std::vector<double> FlexibleBody::computeMassBlock() const {
    // Return diagonal of the mass matrix
    int n = numDof;
    std::vector<double> Mblock(n * n, 0.0);
    // We need const access but getGlobalMassMatrix is non-const; cast away for cache
    auto& self = const_cast<FlexibleBody&>(*this);
    const auto& Mfull = self.getGlobalMassMatrix();
    for (int i = 0; i < n; i++)
        Mblock[i*n+i] = Mfull[i*n+i];
    return Mblock;
}

std::vector<double> FlexibleBody::computeForces(const Vec3& grav) {
    gravity = grav;
    return computeTotalForces();
}

double FlexibleBody::computeKineticEnergy() const {
    // Lumped KE = ½ Σ M_diag_i * v_i²  (consistent with M_diag integrator)
    auto& self = const_cast<FlexibleBody&>(*this);
    const auto& Mdiag = self.getMassDiagonal();
    auto qd = getFlexQd();
    int n = numDof;
    double KE = 0;
    for (int i = 0; i < n; i++)
        KE += Mdiag[i] * qd[i] * qd[i];
    return 0.5 * KE;
}

double FlexibleBody::computePotentialEnergy(const Vec3& grav) const {
    // PE_grav = -Σ M_diag_i * g_d * (q_i - X0_i)  for position DOFs
    auto& self = const_cast<FlexibleBody&>(*this);
    const auto& Mdiag = self.getMassDiagonal();
    double gArr[3] = {grav.x, grav.y, grav.z};
    double PE_grav = 0;
    for (int i = 0; i < (int)nodes.size(); i++) {
        int off = i * ANCF_NODE_DOF;
        for (int d = 0; d < 3; d++) {
            double disp = nodes[i].q[d] - nodes[i].X0[d];
            PE_grav -= Mdiag[off + d] * gArr[d] * disp;
        }
    }
    return computeStrainEnergy() + PE_grav;
}

} // namespace mb
