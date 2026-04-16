#include "mb/fem/FlexibleContactManager.h"
#include <cmath>
#include <algorithm>

namespace mb {

const std::vector<double> FlexibleContactManager::emptyForces_;

FlexibleContactManager::FlexibleContactManager(const FlexContactConfig& config)
    : forceModel_(config)
{
}

void FlexibleContactManager::addBody(FlexibleBody& body, const FlexContactMaterial& mat) {
    auto tris = extractSurfaceTriangles(body);
    auto nodeSet = extractSurfaceNodeIndices(tris);
    bodies_[body.id] = {&body, tris, nodeSet, computeBodyAABB(body, nodeSet), false};
    forceModel_.setBodyMaterial(body.id, mat);
}

void FlexibleContactManager::removeBody(int bodyId) {
    bodies_.erase(bodyId);
    contactForces_.erase(bodyId);
}

void FlexibleContactManager::setBodyMaterial(int bodyId, const FlexContactMaterial& mat) {
    forceModel_.setBodyMaterial(bodyId, mat);
}

void FlexibleContactManager::enableGround(const GroundPlane& ground) {
    groundStorage_ = std::make_unique<GroundPlane>(ground);
    ground_ = groundStorage_.get();
}

void FlexibleContactManager::disableGround() {
    ground_ = nullptr;
    groundStorage_.reset();
}

void FlexibleContactManager::setMinDepth(double d) { detector_.minDepth = d; }
void FlexibleContactManager::setMaxDepth(double d) { detector_.maxDepth = d; }
void FlexibleContactManager::setContactMargin(double d) { detector_.contactMargin = d; }

void FlexibleContactManager::invalidateCache(int bodyId) {
    auto it = bodies_.find(bodyId);
    if (it != bodies_.end()) it->second.dirty = true;
}

void FlexibleContactManager::step() {
    // Rebuild dirty caches
    for (auto& [id, cache] : bodies_) {
        if (cache.dirty) {
            cache.surfaceTris = extractSurfaceTriangles(*cache.body);
            cache.surfaceNodeIndices = extractSurfaceNodeIndices(cache.surfaceTris);
            cache.dirty = false;
        }
    }

    std::vector<FlexContact> allContacts;
    std::vector<BodyCache*> bodyArr;
    for (auto& [id, cache] : bodies_) bodyArr.push_back(&cache);

    // Update AABBs
    for (auto* cache : bodyArr)
        cache->aabb = computeBodyAABB(*cache->body, cache->surfaceNodeIndices);

    double margin = detector_.contactMargin + detector_.maxDepth;

    // ANCF ↔ ANCF (bidirectional)
    for (int i = 0; i < (int)bodyArr.size(); i++) {
        for (int j = i+1; j < (int)bodyArr.size(); j++) {
            auto* A = bodyArr[i];
            auto* B = bodyArr[j];

            if (!aabbOverlap(A->aabb, B->aabb, margin)) continue;

            auto cAB = detector_.detectNodeToSurface(*A->body, A->surfaceNodeIndices, *B->body, B->surfaceTris);
            allContacts.insert(allContacts.end(), cAB.begin(), cAB.end());

            auto cBA = detector_.detectNodeToSurface(*B->body, B->surfaceNodeIndices, *A->body, A->surfaceTris);
            allContacts.insert(allContacts.end(), cBA.begin(), cBA.end());

            if (cAB.empty() && cBA.empty()) {
                auto sat = detector_.detectSAT(*A->body, A->surfaceTris, *B->body, B->surfaceTris);
                allContacts.insert(allContacts.end(), sat.begin(), sat.end());
            }
        }
    }

    // ANCF ↔ Ground
    if (ground_) {
        for (auto* cache : bodyArr) {
            auto cG = detector_.detectNodeToGround(*cache->body, cache->surfaceNodeIndices, *ground_);
            allContacts.insert(allContacts.end(), cG.begin(), cG.end());
        }
    }

    rawContacts_ = allContacts;

    // Force computation
    auto forceResult = forceModel_.computeForces(allContacts);
    activeContacts_ = forceResult.actives;

    // Reaction distribution (barycentric on surface triangles)
    for (const auto& c : allContacts) {
        if (!c.surfBody || c.triIdx < 0) continue;
        auto it = bodies_.find(c.surfBody->id);
        if (it == bodies_.end()) continue;
        auto& cache = it->second;
        if (c.triIdx >= (int)cache.surfaceTris.size()) continue;
        const auto& tri = cache.surfaceTris[c.triIdx];

        auto itF = forceResult.forces.find(c.nodeBody->id);
        if (itF == forceResult.forces.end()) continue;
        int offA = c.nodeIdx * ANCF_NODE_DOF;
        double fx = itF->second[offA], fy = itF->second[offA+1], fz = itF->second[offA+2];

        // Equal distribution (1/3)
        auto& bufB = forceResult.forces[c.surfBody->id];
        if ((int)bufB.size() < c.surfBody->numDof)
            bufB.resize(c.surfBody->numDof, 0.0);
        int triNodes[3] = {tri.n0, tri.n1, tri.n2};
        for (int k = 0; k < 3; k++) {
            int offB = triNodes[k] * ANCF_NODE_DOF;
            bufB[offB]   -= fx / 3.0;
            bufB[offB+1] -= fy / 3.0;
            bufB[offB+2] -= fz / 3.0;
        }
    }

    contactForces_ = std::move(forceResult.forces);
}

const std::vector<double>& FlexibleContactManager::getContactForces(int bodyId) const {
    auto it = contactForces_.find(bodyId);
    return it != contactForces_.end() ? it->second : emptyForces_;
}

} // namespace mb
