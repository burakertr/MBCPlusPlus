#pragma once
#include "mb/contact/ContactTypes.h"
#include <vector>

namespace mb {

/**
 * Computes contact forces for a set of contact pairs.
 * Uses Hertz + Flores normal force model and
 * regularized Coulomb friction model.
 */
class ContactForceModel {
public:
    ContactForceModel();

    void setMaterial(const ContactMaterial& mat) { material_ = mat; }
    const ContactMaterial& getMaterial() const { return material_; }

    /// Compute forces for all contact pairs, return active contacts
    std::vector<ActiveContact> computeForces(
        const std::vector<ContactPair>& pairs, double dt) const;

    /// Compute force for a single pair
    ActiveContact computeSingleForce(const ContactPair& pair, double dt) const;

private:
    ContactMaterial material_;

    /// Hertz + Flores normal force
    double computeNormalForce(double penetration, double vn, double dt) const;

    /// Regularised Coulomb friction
    Vec3 computeFrictionForce(double normalForce, const Vec3& vt) const;
};

} // namespace mb
