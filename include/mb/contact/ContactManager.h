#pragma once
#include "mb/contact/ContactTypes.h"
#include "mb/contact/CollisionDetector.h"
#include "mb/contact/ContactForceModel.h"
#include "mb/core/Body.h"
#include <vector>
#include <memory>
#include <unordered_map>

namespace mb {

/**
 * Manages the full contact pipeline:
 *   1. Collision detection
 *   2. Contact force computation
 *   3. Force application to bodies
 *   4. Impulse-based position/velocity correction
 */
class ContactManager {
public:
    ContactManager();

    void setConfig(const ContactConfig& config) { config_ = config; }
    const ContactConfig& getConfig() const { return config_; }

    void setMaterial(const ContactMaterial& mat);
    const ContactMaterial& getMaterial() const;

    /// Run the full detect → compute → apply pipeline
    void processContacts(const std::vector<std::shared_ptr<Body>>& bodies, double dt);

    /// Apply impulse-based correction for penetration
    void resolveImpulses(const std::vector<std::shared_ptr<Body>>& bodies, double dt);

    /// Get currently active contacts
    const std::vector<ActiveContact>& getActiveContacts() const { return activeContacts_; }

    /// Number of active contacts
    int getContactCount() const { return static_cast<int>(activeContacts_.size()); }

    /// Total contact force magnitude
    double getTotalContactForce() const;

    /// Clear all contact state
    void clear();

private:
    ContactConfig config_;
    CollisionDetector detector_;
    ContactForceModel forceModel_;
    std::vector<ActiveContact> activeContacts_;

    /// Apply resolved forces to bodies
    void applyContactForces();

    /// Match new contacts with existing ones for warm-starting
    void updateContactLifetimes(const std::vector<ContactPair>& newPairs);
};

} // namespace mb
