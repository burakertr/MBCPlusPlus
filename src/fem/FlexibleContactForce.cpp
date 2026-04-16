#include "mb/fem/FlexibleContactForce.h"
#include <cmath>
#include <algorithm>

namespace mb {

static const FlexContactMaterial DEFAULT_MAT;

FlexibleContactForce::FlexibleContactForce(const FlexContactConfig& config)
    : config_(config) {}

void FlexibleContactForce::setBodyMaterial(int bodyId, const FlexContactMaterial& mat) {
    materialMap_[bodyId] = mat;
}

const FlexContactMaterial& FlexibleContactForce::getMaterial(int bodyId) const {
    auto it = materialMap_.find(bodyId);
    return it != materialMap_.end() ? it->second : DEFAULT_MAT;
}

double FlexibleContactForce::combinedModulus(double E1, double nu1, double E2, double nu2) {
    double inv1 = (1.0 - nu1*nu1) / E1;
    double inv2 = (1.0 - nu2*nu2) / E2;
    return 1.0 / (inv1 + inv2);
}

int64_t FlexibleContactForce::contactKey(int bodyAId, int nodeIdx, int surfBodyId) {
    return (int64_t)bodyAId * 10000000LL + (int64_t)nodeIdx * 1000LL + (int64_t)(surfBodyId + 1);
}

const std::vector<double>& FlexibleContactForce::getNodeRadii(FlexibleBody& body) {
    auto it = nodeRadiusCache_.find(body.id);
    if (it != nodeRadiusCache_.end() && (int)it->second.size() == (int)body.nodes.size())
        return it->second;

    int nNodes = (int)body.nodes.size();
    std::vector<double> radii(nNodes, 0.01);

    // Build node→element adjacency
    std::vector<std::vector<int>> adj(nNodes);
    for (int ei = 0; ei < (int)body.elements.size(); ei++)
        for (int ni : body.elements[ei].nodeIds)
            adj[ni].push_back(ei);

    for (int ni = 0; ni < nNodes; ni++) {
        double px = body.nodes[ni].q[0], py = body.nodes[ni].q[1], pz = body.nodes[ni].q[2];
        double sumLen = 0; int count = 0;
        for (int ei : adj[ni]) {
            for (int oj : body.elements[ei].nodeIds) {
                if (oj == ni) continue;
                double dx = body.nodes[oj].q[0]-px, dy = body.nodes[oj].q[1]-py, dz = body.nodes[oj].q[2]-pz;
                sumLen += std::sqrt(dx*dx + dy*dy + dz*dz);
                count++;
            }
        }
        radii[ni] = count > 0 ? (sumLen/count) * 0.5 : 0.01;
    }

    nodeRadiusCache_[body.id] = std::move(radii);
    return nodeRadiusCache_[body.id];
}

void FlexibleContactForce::invalidateRadiusCache(int bodyId) {
    nodeRadiusCache_.erase(bodyId);
}

FlexibleContactForce::ForceResult FlexibleContactForce::computeForces(
    const std::vector<FlexContact>& contacts)
{
    ForceResult result;
    std::set<int64_t> currentContacts;

    auto ensureBuf = [&](FlexibleBody* body) -> std::vector<double>& {
        auto it = result.forces.find(body->id);
        if (it == result.forces.end()) {
            result.forces[body->id] = std::vector<double>(body->numDof, 0.0);
            return result.forces[body->id];
        }
        return it->second;
    };

    // Pre-pass: track impact velocities
    for (const auto& c : contacts) {
        int surfId = c.surfBody ? c.surfBody->id : -1;
        int64_t key = contactKey(c.nodeBody->id, c.nodeIdx, surfId);
        currentContacts.insert(key);
        if (previousContacts_.find(key) == previousContacts_.end()) {
            auto& qd = c.nodeBody->nodes[c.nodeIdx].qd;
            double vn = qd[0]*c.normal.x + qd[1]*c.normal.y + qd[2]*c.normal.z;
            impactVelocities_[key] = std::abs(std::min(vn, -1e-6));
        }
    }

    double hertzN = config_.hertzExponent;
    double vReg = config_.frictionRegVelocity;
    double maxK = config_.maxStiffness;
    double cfgRadius = config_.contactRadius;

    for (const auto& c : contacts) {
        const auto& matA = getMaterial(c.nodeBody->id);
        const auto& matB = c.surfBody ? getMaterial(c.surfBody->id) : DEFAULT_MAT;

        double e = std::sqrt(matA.restitution * matB.restitution);
        double mu = std::sqrt(matA.friction * matB.friction);
        double Estar = combinedModulus(matA.youngsModulus, matA.poissonRatio,
                                       matB.youngsModulus, matB.poissonRatio);

        double Rstar = cfgRadius;
        if (Rstar <= 0) {
            const auto& radii = getNodeRadii(*c.nodeBody);
            Rstar = radii[c.nodeIdx];
        }

        double K = (4.0/3.0) * Estar * std::sqrt(std::max(Rstar, 1e-6));
        if (K > maxK) K = maxK;

        double delta = std::max(c.depth, 0.0);
        double deltaN = (hertzN == 1.5) ? delta * std::sqrt(delta) : std::pow(delta, hertzN);

        auto& qd = c.nodeBody->nodes[c.nodeIdx].qd;
        double vrx = qd[0], vry = qd[1], vrz = qd[2];
        double nx = c.normal.x, ny = c.normal.y, nz = c.normal.z;
        double vn = vrx*nx + vry*ny + vrz*nz;

        int surfId = c.surfBody ? c.surfBody->id : -1;
        int64_t key = contactKey(c.nodeBody->id, c.nodeIdx, surfId);
        double vImpact = std::max(impactVelocities_.count(key) ? impactVelocities_[key] : 1e-3, 0.01);

        double deltaDot = -vn;
        double D = 0;
        if (e < 1 && vImpact > 1e-8)
            D = 8.0*(1.0-e) / (5.0*e*vImpact);
        double dampMul = 1.0 + D * deltaDot;
        dampMul = std::clamp(dampMul, 0.0, 5.0);
        double Fn = K * deltaN * dampMul;
        if (Fn < 0) Fn = 0;

        // Friction
        double vtx = vrx - vn*nx, vty = vry - vn*ny, vtz = vrz - vn*nz;
        double vtMag = std::sqrt(vtx*vtx + vty*vty + vtz*vtz);
        double ffx = 0, ffy = 0, ffz = 0;
        if (vtMag > 1e-12 && mu > 0) {
            double scale = mu * Fn / (vtMag > vReg ? vtMag : vReg);
            ffx = -vtx*scale; ffy = -vty*scale; ffz = -vtz*scale;
        }

        double tfx = Fn*nx + ffx, tfy = Fn*ny + ffy, tfz = Fn*nz + ffz;

        if (tfx*tfx + tfy*tfy + tfz*tfz > 1e-30) {
            auto& buf = ensureBuf(c.nodeBody);
            int off = c.nodeIdx * ANCF_NODE_DOF;
            buf[off] += tfx; buf[off+1] += tfy; buf[off+2] += tfz;
        }

        result.actives.push_back({
            c.nodeBody, c.nodeIdx, c.surfBody, c.point, c.normal, delta,
            Fn, Vec3(ffx, ffy, ffz), Vec3(tfx, tfy, tfz)
        });
    }

    // Clean up stale impact velocities
    for (auto it = impactVelocities_.begin(); it != impactVelocities_.end(); ) {
        if (currentContacts.find(it->first) == currentContacts.end())
            it = impactVelocities_.erase(it);
        else ++it;
    }
    previousContacts_ = currentContacts;

    return result;
}

} // namespace mb
