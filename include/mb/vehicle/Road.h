#pragma once
#include "mb/math/Vec3.h"
#include <vector>
#include <string>

namespace mb {

/**
 * Abstract road / terrain interface (Y is up).
 * height(x,z) returns ground elevation y at planar location (x,z).
 * normal(x,z) returns outward unit normal at that point.
 * friction(x,z) returns the local friction coefficient (μ).
 */
class Road {
public:
    virtual ~Road() = default;

    /// Ground elevation y at planar (x,z).
    virtual double height(double x, double z) const = 0;

    /// Outward unit normal at (x,z).
    virtual Vec3 normal(double x, double z) const = 0;

    /// Local friction coefficient at (x,z).
    virtual double friction(double x, double z) const = 0;

    /// Convenience: signed penetration depth of a sphere (centre, radius).
    /// Returns max(0, radius - distanceFromGroundAlongNormal). Also outputs
    /// the contact point and surface normal at the in-plane projection.
    double penetration(const Vec3& centre, double radius,
                       Vec3* contactPoint = nullptr,
                       Vec3* contactNormal = nullptr) const;
};

/**
 * Flat horizontal road at y = elevation, with constant friction.
 */
class FlatRoad : public Road {
public:
    explicit FlatRoad(double elevation = 0.0, double mu = 1.0)
        : elevation_(elevation), mu_(mu) {}

    double height(double, double) const override { return elevation_; }
    Vec3 normal(double, double) const override { return {0.0, 1.0, 0.0}; }
    double friction(double, double) const override { return mu_; }

    void setFriction(double mu) { mu_ = mu; }
    void setElevation(double y) { elevation_ = y; }

private:
    double elevation_;
    double mu_;
};

/**
 * Heightmap road: regular 2D grid of elevations sampled in (x,z) plane.
 *   x = originX + ix * dx,  z = originZ + iz * dz
 * Bilinear interpolation for height; central differences for normal.
 */
class HeightmapRoad : public Road {
public:
    HeightmapRoad(int nx, int nz,
                  double dx, double dz,
                  double originX, double originZ,
                  std::vector<double> heights,
                  double mu = 1.0);

    /// Procedural sinusoidal terrain (useful for tests/demos).
    static HeightmapRoad makeSinusoidal(int nx, int nz,
                                        double dx, double dz,
                                        double originX, double originZ,
                                        double amplitude, double wavelength,
                                        double mu = 1.0);

    double height(double x, double z) const override;
    Vec3 normal(double x, double z) const override;
    double friction(double, double) const override { return mu_; }

    void setFriction(double mu) { mu_ = mu; }

private:
    int nx_, nz_;
    double dx_, dz_;
    double originX_, originZ_;
    std::vector<double> h_; // size nx_ * nz_, indexed (iz * nx_ + ix)
    double mu_;

    double sample(int ix, int iz) const;
};

} // namespace mb
