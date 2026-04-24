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

    // Build elements (tet4 + hex8)
    for (const auto& ge : mesh.elements) {
        if (ge.type == 4) {
            if (ge.nodeIds.size() != 4)
                throw std::runtime_error("Tet element has invalid node count");

            TetConnectivity conn;
            for (int k = 0; k < 4; k++) {
                auto it = nodeIdMap.find(ge.nodeIds[k]);
                if (it == nodeIdMap.end())
                    throw std::runtime_error("Node " + std::to_string(ge.nodeIds[k]) + " not found");
                conn.nodeIds[k] = it->second;
            }
            body->elements.emplace_back(conn, body->nodes, *body->material, highOrderQuad);
        } else if (ge.type == 5) {
            if (ge.nodeIds.size() != 8)
                throw std::runtime_error("Hex element has invalid node count");

            HexConnectivity conn;
            for (int k = 0; k < 8; k++) {
                auto it = nodeIdMap.find(ge.nodeIds[k]);
                if (it == nodeIdMap.end())
                    throw std::runtime_error("Node " + std::to_string(ge.nodeIds[k]) + " not found");
                conn.nodeIds[k] = it->second;
            }
            body->hexElements.emplace_back(conn, body->nodes, *body->material, highOrderQuad);
        }
    }

    // Full ANCF formulation: gradient DOFs (3..11) are FREE.
    // The deformation gradient F is interpolated from gradient DOFs
    // at each node, and elastic forces act on all 48 DOFs per element.
    // Gradient DOFs are initialized to F = I (identity) in the
    // undeformed configuration (already set above).

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

    for (const auto& elem : hexElements) {
        if (!elem.alive) continue;
        auto Me = elem.computeMassMatrix(nodes);
        const auto& nids = elem.nodeIds;

        for (int a = 0; a < 8; a++)
            for (int b = 0; b < 8; b++) {
                int gA = nids[a] * ANCF_NODE_DOF;
                int gB = nids[b] * ANCF_NODE_DOF;
                for (int i = 0; i < ANCF_NODE_DOF; i++)
                    for (int j = 0; j < ANCF_NODE_DOF; j++) {
                        int eRow = a * ANCF_NODE_DOF + i;
                        int eCol = b * ANCF_NODE_DOF + j;
                        globalMass_[(gA+i)*n + (gB+j)] += Me[eRow*ANCF_HEX_ELEM_DOF+eCol];
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

void FlexibleBody::syncGradientDOFs() {
    int nNodes = (int)nodes.size();
    std::vector<double> Faccum(nNodes * 9, 0.0);
    std::vector<double> Vaccum(nNodes, 0.0);

    for (const auto& elem : elements) {
        if (!elem.alive) continue;
        double Felem[9];
        elem.computePositionBasedF(nodes, Felem);
        double vol = elem.V0;
        for (int k = 0; k < 4; k++) {
            int nid = elem.nodeIds[k];
            Vaccum[nid] += vol;
            for (int d = 0; d < 9; d++)
                Faccum[nid * 9 + d] += vol * Felem[d];
        }
    }

    for (const auto& elem : hexElements) {
        if (!elem.alive) continue;
        double Felem[9];
        elem.computePositionBasedF(nodes, Felem);
        double vol = elem.V0;
        for (int k = 0; k < 8; k++) {
            int nid = elem.nodeIds[k];
            Vaccum[nid] += vol;
            for (int d = 0; d < 9; d++)
                Faccum[nid * 9 + d] += vol * Felem[d];
        }
    }

    for (int ni = 0; ni < nNodes; ni++) {
        if (Vaccum[ni] < 1e-30) continue;
        double invV = 1.0 / Vaccum[ni];
        for (int d = 0; d < 9; d++)
            nodes[ni].q[3 + d] = Faccum[ni * 9 + d] * invV;
    }
}

std::vector<double> FlexibleBody::computeElasticForces() {
    int n = numDof;
    std::vector<double> Q(n, 0.0);
    int nElem = (int)elements.size();
    int nHex = (int)hexElements.size();

    if (positionOnlyMode) {
        // Position-only: compute elastic forces directly from position DOFs.
        // Uses nodal-averaged deformation gradient (F-bar approach) to
        // mitigate volumetric locking in constant-strain linear tetrahedra.
        //
        // F-bar method: replace the volumetric part of the element F with
        // a nodal-averaged J, so that:
        //   F̄ = (J̄/J)^{1/3} · F
        // where J̄ is the volume-weighted average of J at the element's nodes.
        //
        // For elements with ν close to 0.5 or stiff volumetric response,
        // this dramatically reduces volumetric locking and allows the
        // constant-strain tet mesh to represent bending correctly.

        // Step 1: Compute per-element F and J
        std::vector<double> elemF(nElem * 9);
        std::vector<double> elemJ(nElem);
        for (int ei = 0; ei < nElem; ei++) {
            if (!elements[ei].alive) continue;
            elements[ei].computePositionBasedF(nodes, &elemF[ei * 9]);
            double* F = &elemF[ei * 9];
            elemJ[ei] = F[0]*(F[4]*F[8]-F[5]*F[7])
                      - F[1]*(F[3]*F[8]-F[5]*F[6])
                      + F[2]*(F[3]*F[7]-F[4]*F[6]);
        }

        // Step 2: Compute nodal-averaged J (volume-weighted)
        int nNodes = (int)nodes.size();
        std::vector<double> Jsum(nNodes, 0.0);
        std::vector<double> Vsum(nNodes, 0.0);
        for (int ei = 0; ei < nElem; ei++) {
            if (!elements[ei].alive) continue;
            double vol = elements[ei].V0;
            for (int a = 0; a < 4; a++) {
                int nid = elements[ei].nodeIds[a];
                Jsum[nid] += vol * elemJ[ei];
                Vsum[nid] += vol;
            }
        }
        std::vector<double> Javg_node(nNodes, 1.0);
        for (int ni = 0; ni < nNodes; ni++) {
            if (Vsum[ni] > 1e-30)
                Javg_node[ni] = Jsum[ni] / Vsum[ni];
        }

        // Step 3: Compute elastic forces using F-bar
        #pragma omp parallel
        {
            std::vector<double> Qloc(n, 0.0);
            #pragma omp for schedule(dynamic) nowait
            for (int ei = 0; ei < nElem; ei++) {
                if (!elements[ei].alive) continue;
                const auto& elem = elements[ei];
                const auto& nids = elem.nodeIds;

                double* F = &elemF[ei * 9];
                double J = elemJ[ei];

                // Compute element-level averaged J from its nodes
                double Javg = 0;
                for (int a = 0; a < 4; a++)
                    Javg += 0.25 * Javg_node[nids[a]];

                // F-bar scaling: (Javg/J)^{1/3}
                double Fbar[9];
                if (std::abs(J) > 1e-30) {
                    double scale = std::cbrt(Javg / J);
                    for (int i = 0; i < 9; i++)
                        Fbar[i] = scale * F[i];
                } else {
                    for (int i = 0; i < 9; i++)
                        Fbar[i] = F[i];
                }

                double P[9];
                material->firstPiolaStress(Fbar, P);

                double factor = elem.V0;

                for (int a = 0; a < 4; a++) {
                    int gOff = nids[a] * ANCF_NODE_DOF;
                    for (int k = 0; k < 3; k++) {
                        double val = 0;
                        for (int j = 0; j < 3; j++)
                            val += P[k*3+j] * elem.dNdX_[a][j];
                        Qloc[gOff + k] -= val * factor;
                    }
                }
            }
            #pragma omp critical
            for (int i = 0; i < n; i++) Q[i] += Qloc[i];
        }

        #pragma omp parallel
        {
            std::vector<double> Qloc(n, 0.0);
            #pragma omp for schedule(dynamic) nowait
            for (int ei = 0; ei < nHex; ei++) {
                if (!hexElements[ei].alive) continue;
                const auto& elem = hexElements[ei];
                auto Qe = elem.computeElasticForcesPositionOnly(nodes);
                const auto& nids = elem.nodeIds;
                for (int a = 0; a < 8; a++) {
                    int gOff = nids[a] * ANCF_NODE_DOF;
                    for (int d = 0; d < ANCF_NODE_DOF; d++)
                        Qloc[gOff+d] += Qe[a*ANCF_NODE_DOF+d];
                }
            }
            #pragma omp critical
            for (int i = 0; i < n; i++) Q[i] += Qloc[i];
        }
    } else {
        // Full ANCF mode (original)
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

        #pragma omp parallel
        {
            std::vector<double> Qloc(n, 0.0);
            #pragma omp for schedule(dynamic) nowait
            for (int ei = 0; ei < nHex; ei++) {
                if (!hexElements[ei].alive) continue;
                const auto& elem = hexElements[ei];
                auto Qe = elem.computeElasticForces(nodes);
                const auto& nids = elem.nodeIds;
                for (int a = 0; a < 8; a++) {
                    int gOff = nids[a] * ANCF_NODE_DOF;
                    for (int d = 0; d < ANCF_NODE_DOF; d++)
                        Qloc[gOff+d] += Qe[a*ANCF_NODE_DOF+d];
                }
            }
            #pragma omp critical
            for (int i = 0; i < n; i++) Q[i] += Qloc[i];
        }
    }

    for (int d = 0; d < n; d++)
        if (fixedDofMask_[d]) Q[d] = 0;

    return Q;
}

std::vector<double> FlexibleBody::computeGravityForces() {
    int n = numDof;
    std::vector<double> Q(n, 0.0);
    double gVec[3] = {gravity.x, gravity.y, gravity.z};

    // Consistent nodal gravity forces via element-level lumped mass.
    // Tet: each node of element gets ρ·V0/4·g; Hex: ρ·V0/8·g.
    // Gravity is applied to ALL nodes including constrained ones so that
    // Σ Q[..+d] = ρ·V_total·g exactly (no mass is silently dropped).
    // The solver enforces displacement constraints via freeIdx partitioning;
    // fixed-DOF gravity entries become wall reaction contributions and do
    // not affect the free-DOF displacement solution.
    double rho = material->density();

    // Gravity is applied to ALL nodes regardless of constraints.
    // Fixed DOF entries in Q are ignored by the static/dynamic solver
    // (the solver only reads freeIdx entries), so they correctly become
    // wall reaction forces without affecting the displacement solution.
    for (const auto& elem : elements) {
        if (!elem.alive) continue;
        double nodalForce = rho * elem.V0 / 4.0;
        for (int a = 0; a < 4; a++) {
            int gOff = elem.nodeIds[a] * ANCF_NODE_DOF;
            for (int d = 0; d < 3; d++)
                Q[gOff + d] += nodalForce * gVec[d];
        }
    }

    for (const auto& elem : hexElements) {
        if (!elem.alive) continue;
        double nodalForce = rho * elem.V0 / 8.0;
        for (int a = 0; a < 8; a++) {
            int gOff = elem.nodeIds[a] * ANCF_NODE_DOF;
            for (int d = 0; d < 3; d++)
                Q[gOff + d] += nodalForce * gVec[d];
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

    // ── ANCF Gradient Consistency ─────────────────────────────
    // In position-only mode, gradient DOFs are slaves — no penalty needed.
    // In full ANCF mode, apply penalty to couple gradient DOFs to positions.
    if (!positionOnlyMode) {
        double kappa = gradientPenalty;
        if (kappa <= 0) kappa = materialProps.E;

        int nNodes = (int)nodes.size();
        std::vector<double> Faccum(nNodes * 9, 0.0);
        std::vector<double> Vaccum(nNodes, 0.0);

        for (const auto& elem : elements) {
            if (!elem.alive) continue;
            double Felem[9];
            elem.computePositionBasedF(nodes, Felem);
            double vol = elem.V0;
            for (int k = 0; k < 4; k++) {
                int nid = elem.nodeIds[k];
                Vaccum[nid] += vol;
                for (int d = 0; d < 9; d++)
                    Faccum[nid * 9 + d] += vol * Felem[d];
            }
        }

        for (const auto& elem : hexElements) {
            if (!elem.alive) continue;
            double Felem[9];
            elem.computePositionBasedF(nodes, Felem);
            double vol = elem.V0;
            for (int k = 0; k < 8; k++) {
                int nid = elem.nodeIds[k];
                Vaccum[nid] += vol;
                for (int d = 0; d < 9; d++)
                    Faccum[nid * 9 + d] += vol * Felem[d];
            }
        }

        #pragma omp parallel for schedule(static)
        for (int ni = 0; ni < nNodes; ni++) {
            if (Vaccum[ni] < 1e-30) continue;
            int off = ni * ANCF_NODE_DOF;
            double invV = 1.0 / Vaccum[ni];
            for (int d = 0; d < 9; d++) {
                int gDof = off + 3 + d;
                if (!fixedDofMask_[gDof]) {
                    double Ftarget = Faccum[ni * 9 + d] * invV;
                    double Fcurrent = nodes[ni].q[3 + d];
                    Q[gDof] -= kappa * (Fcurrent - Ftarget);
                }
            }
        }
    }

    // Zero fixed DOFs
    for (int i = 0; i < n; i++)
        if (fixedDofMask_[i]) Q[i] = 0;

    return Q;
}

// ─── Stiffness Matrix Assembly ───────────────────────────────

std::vector<double> FlexibleBody::assembleStiffnessMatrix() {
    int n = numDof;
    std::vector<double> K(n * n, 0.0);
    int nElem = (int)elements.size();
    int nHex = (int)hexElements.size();

    if (positionOnlyMode) {
        // Position-only mode: assemble only pos-pos block directly
        // K(a_k, b_m) = Σ_jl dNa/dXj · dPdF(kj,ml) · dNb/dXl · V0
        // This is -dQe/dq (positive definite)
        #pragma omp parallel
        {
            std::vector<double> Kloc(n * n, 0.0);
            #pragma omp for schedule(dynamic) nowait
            for (int ei = 0; ei < nElem; ei++) {
                if (!elements[ei].alive) continue;
                const auto& elem = elements[ei];
                const auto& nids = elem.nodeIds;

                double F[9];
                elem.computePositionBasedF(nodes, F);

                double dPdF[81];
                elem.computeMaterialTangent(F, dPdF);

                double factor = elem.V0;

                for (int a = 0; a < 4; a++) {
                    for (int b = 0; b < 4; b++) {
                        int gA = nids[a] * ANCF_NODE_DOF;
                        int gB = nids[b] * ANCF_NODE_DOF;
                        for (int m1 = 0; m1 < 3; m1++) {
                            for (int m2 = 0; m2 < 3; m2++) {
                                double val = 0;
                                for (int j = 0; j < 3; j++)
                                    for (int l = 0; l < 3; l++)
                                        val += elem.dNdX_[a][j] * dPdF[(m1*3+j)*9 + (m2*3+l)] * elem.dNdX_[b][l];
                                Kloc[(gA+m1)*n + (gB+m2)] += val * factor;
                            }
                        }
                    }
                }
            }
            #pragma omp critical
            for (int i = 0; i < n * n; i++) K[i] += Kloc[i];
        }

        #pragma omp parallel
        {
            std::vector<double> Kloc(n * n, 0.0);
            #pragma omp for schedule(dynamic) nowait
            for (int ei = 0; ei < nHex; ei++) {
                if (!hexElements[ei].alive) continue;
                const auto& elem = hexElements[ei];
                auto Ke = elem.computeStiffnessMatrixPositionOnly(nodes);
                const auto& nids = elem.nodeIds;

                for (int a = 0; a < 8; a++)
                    for (int b = 0; b < 8; b++) {
                        int gA = nids[a] * ANCF_NODE_DOF;
                        int gB = nids[b] * ANCF_NODE_DOF;
                        for (int i = 0; i < ANCF_NODE_DOF; i++)
                            for (int j = 0; j < ANCF_NODE_DOF; j++) {
                                int eRow = a * ANCF_NODE_DOF + i;
                                int eCol = b * ANCF_NODE_DOF + j;
                                Kloc[(gA+i)*n + (gB+j)] += Ke[eRow*ANCF_HEX_ELEM_DOF+eCol];
                            }
                    }
            }
            #pragma omp critical
            for (int i = 0; i < n * n; i++) K[i] += Kloc[i];
        }
    } else {
        // Full ANCF mode (original)
        #pragma omp parallel
        {
            std::vector<double> Kloc(n * n, 0.0);
            #pragma omp for schedule(dynamic) nowait
            for (int ei = 0; ei < nElem; ei++) {
                if (!elements[ei].alive) continue;
                const auto& elem = elements[ei];
                auto Ke = elem.computeStiffnessMatrix(nodes);
                const auto& nids = elem.nodeIds;

                for (int a = 0; a < 4; a++)
                    for (int b = 0; b < 4; b++) {
                        int gA = nids[a] * ANCF_NODE_DOF;
                        int gB = nids[b] * ANCF_NODE_DOF;
                        for (int i = 0; i < ANCF_NODE_DOF; i++)
                            for (int j = 0; j < ANCF_NODE_DOF; j++) {
                                int eRow = a * ANCF_NODE_DOF + i;
                                int eCol = b * ANCF_NODE_DOF + j;
                                Kloc[(gA+i)*n + (gB+j)] += Ke[eRow*48+eCol];
                            }
                    }
            }
            #pragma omp critical
            for (int i = 0; i < n * n; i++) K[i] += Kloc[i];
        }

        #pragma omp parallel
        {
            std::vector<double> Kloc(n * n, 0.0);
            #pragma omp for schedule(dynamic) nowait
            for (int ei = 0; ei < nHex; ei++) {
                if (!hexElements[ei].alive) continue;
                const auto& elem = hexElements[ei];
                auto Ke = elem.computeStiffnessMatrix(nodes);
                const auto& nids = elem.nodeIds;

                for (int a = 0; a < 8; a++)
                    for (int b = 0; b < 8; b++) {
                        int gA = nids[a] * ANCF_NODE_DOF;
                        int gB = nids[b] * ANCF_NODE_DOF;
                        for (int i = 0; i < ANCF_NODE_DOF; i++)
                            for (int j = 0; j < ANCF_NODE_DOF; j++) {
                                int eRow = a * ANCF_NODE_DOF + i;
                                int eCol = b * ANCF_NODE_DOF + j;
                                Kloc[(gA+i)*n + (gB+j)] += Ke[eRow*ANCF_HEX_ELEM_DOF+eCol];
                            }
                    }
            }
            #pragma omp critical
            for (int i = 0; i < n * n; i++) K[i] += Kloc[i];
        }

        // Add gradient consistency penalty stiffness: κ·I on gradient DOFs
        double kappa = gradientPenalty;
        if (kappa <= 0) kappa = materialProps.E;

        int nNodes = (int)nodes.size();
        for (int ni = 0; ni < nNodes; ni++) {
            int off = ni * ANCF_NODE_DOF;
            for (int d = 3; d < ANCF_NODE_DOF; d++) {
                int gDof = off + d;
                if (!fixedDofMask_[gDof])
                    K[gDof * n + gDof] += kappa;
            }
        }
    }

    // Zero rows/cols for fixed DOFs
    for (int d = 0; d < n; d++) {
        if (fixedDofMask_[d]) {
            for (int j = 0; j < n; j++) {
                K[d*n+j] = 0;
                K[j*n+d] = 0;
            }
        }
    }

    return K;
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

    int nHex = (int)hexElements.size();
    #pragma omp parallel for reduction(+:W) schedule(dynamic)
    for (int ei = 0; ei < nHex; ei++) {
        if (!hexElements[ei].alive) continue;
        double F[9];
        hexElements[ei].computeDeformationGradient(0.0, 0.0, 0.0, nodes, F);
        W += material->strainEnergyDensity(F) * hexElements[ei].V0;
    }

    return W;
}

double FlexibleBody::computeElementVonMises(int elemIdx) const {
    if (elemIdx < 0) return 0;

    bool isTet = elemIdx < (int)elements.size();
    int localIdx = elemIdx;
    if (!isTet) localIdx -= (int)elements.size();

    if (isTet) {
        if (localIdx >= (int)elements.size()) return 0;
        if (!elements[localIdx].alive) return 0;
    } else {
        if (localIdx >= (int)hexElements.size()) return 0;
        if (!hexElements[localIdx].alive) return 0;
    }

    // Deformation gradient at centroid
    double F[9];
    if (isTet)
        elements[localIdx].computeDeformationGradient(0.25, 0.25, 0.25, nodes, F);
    else
        hexElements[localIdx].computeDeformationGradient(0.0, 0.0, 0.0, nodes, F);

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

    int nTet = (int)elements.size();
    for (int idx : elemIndices) {
        if (idx >= 0 && idx < nTet) {
            elements[idx].alive = false;
        } else {
            int h = idx - nTet;
            if (h >= 0 && h < (int)hexElements.size())
                hexElements[h].alive = false;
        }
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
    for (const auto& elem : hexElements)
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

std::vector<std::array<int,8>> FlexibleBody::getHexConnectivity() const {
    std::vector<std::array<int,8>> conn;
    conn.reserve(hexElements.size());
    for (const auto& e : hexElements)
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
