#pragma once
#include "mb/math/Vec3.h"
#include "mb/fem/FlexibleBody.h"
#include "mb/fem/FlexibleContactDetector.h"
#include "mb/fem/ANCFTypes.h"
#include <vector>
#include <map>
#include <set>

namespace mb {

// ─── Material & config ───────────────────────────────────────

struct FlexContactMaterial {
    double restitution = 0.5;
    double friction = 0.4;
    double youngsModulus = 5e5;
    double poissonRatio = 0.3;
};

struct FlexContactConfig {
    double hertzExponent = 1.5;
    double frictionRegVelocity = 1e-3;
    double maxStiffness = 1e6;
    double contactRadius = 0;   ///< 0 = auto-estimate
};

// ─── Active contact diagnostics ──────────────────────────────

struct FlexActiveContact {
    FlexibleBody* nodeBody;
    int nodeIdx;
    FlexibleBody* surfBody;
    Vec3 point;
    Vec3 normal;
    double depth;
    double normalForce;
    Vec3 frictionForce;
    Vec3 totalForce;
};

// ─── Force model ─────────────────────────────────────────────

class FlexibleContactForce {
public:
    FlexibleContactForce(const FlexContactConfig& config = {});

    void setBodyMaterial(int bodyId, const FlexContactMaterial& mat);

    struct ForceResult {
        std::map<int, std::vector<double>> forces;  ///< bodyId → force array
        std::vector<FlexActiveContact> actives;
    };

    ForceResult computeForces(const std::vector<FlexContact>& contacts);

    void invalidateRadiusCache(int bodyId);

private:
    FlexContactConfig config_;
    std::map<int, FlexContactMaterial> materialMap_;
    std::map<int64_t, double> impactVelocities_;
    std::set<int64_t> previousContacts_;
    std::map<int, std::vector<double>> nodeRadiusCache_;

    const FlexContactMaterial& getMaterial(int bodyId) const;
    const std::vector<double>& getNodeRadii(FlexibleBody& body);

    static double combinedModulus(double E1, double nu1, double E2, double nu2);
    static int64_t contactKey(int bodyAId, int nodeIdx, int surfBodyId);
};

} // namespace mb
