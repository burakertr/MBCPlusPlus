#pragma once
#include "mb/math/Vec3.h"
#include "mb/fem/FlexibleBody.h"
#include "mb/fem/FlexibleContactDetector.h"
#include "mb/fem/FlexibleContactForce.h"
#include <vector>
#include <map>
#include <set>
#include <memory>

namespace mb {

/**
 * Flexible Contact Manager — Orchestrator.
 *
 * Full pipeline: cache surfaces → AABB broad-phase → narrow-phase
 * detection → Hertz+Flores+Coulomb forces → reaction distribution.
 */
class FlexibleContactManager {
public:
    double maxForcePerDof = 500.0;

    FlexibleContactManager(const FlexContactConfig& config = {});

    void addBody(FlexibleBody& body, const FlexContactMaterial& mat = {});
    void removeBody(int bodyId);
    void setBodyMaterial(int bodyId, const FlexContactMaterial& mat);

    void enableGround(const GroundPlane& ground);
    void disableGround();

    void setMinDepth(double d);
    void setMaxDepth(double d);
    void setContactMargin(double d);

    void invalidateCache(int bodyId);

    /// Run full contact pipeline for one time step
    void step();

    /// Get contact forces for a body (empty if no contacts)
    const std::vector<double>& getContactForces(int bodyId) const;

    bool hasContacts() const { return !rawContacts_.empty(); }
    int numContacts() const { return (int)rawContacts_.size(); }
    const std::vector<FlexActiveContact>& activeContacts() const { return activeContacts_; }
    const std::vector<FlexContact>& rawContacts() const { return rawContacts_; }

private:
    struct BodyCache {
        FlexibleBody* body;
        std::vector<SurfaceTriangle> surfaceTris;
        std::set<int> surfaceNodeIndices;
        AABB aabb;
        bool dirty = false;
    };

    FlexibleContactDetector detector_;
    FlexibleContactForce forceModel_;

    std::map<int, BodyCache> bodies_;
    GroundPlane* ground_ = nullptr;
    std::unique_ptr<GroundPlane> groundStorage_;

    std::map<int, std::vector<double>> contactForces_;
    std::vector<FlexActiveContact> activeContacts_;
    std::vector<FlexContact> rawContacts_;
    static const std::vector<double> emptyForces_;
};

} // namespace mb
