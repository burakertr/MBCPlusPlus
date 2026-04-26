#include "mb/vehicle/Road.h"
#include <algorithm>
#include <cmath>

namespace mb {

double Road::penetration(const Vec3& centre, double radius,
                         Vec3* contactPoint, Vec3* contactNormal) const {
    double y = height(centre.x, centre.z);
    Vec3 n = normal(centre.x, centre.z);
    // Signed perpendicular distance from sphere centre to ground plane at (x,z),
    // approximated as (centre.y - y) projected on normal. For mild slopes this
    // is essentially centre.y - y; for completeness we use the proper proj.
    Vec3 ground{centre.x, y, centre.z};
    double signedDist = centre.sub(ground).dot(n);
    double pen = radius - signedDist;
    if (contactPoint)  *contactPoint  = centre.sub(n.scale(signedDist));
    if (contactNormal) *contactNormal = n;
    return pen > 0.0 ? pen : 0.0;
}

// ---------------- HeightmapRoad ----------------

HeightmapRoad::HeightmapRoad(int nx, int nz,
                             double dx, double dz,
                             double originX, double originZ,
                             std::vector<double> heights,
                             double mu)
    : nx_(nx), nz_(nz), dx_(dx), dz_(dz),
      originX_(originX), originZ_(originZ),
      h_(std::move(heights)), mu_(mu) {}

HeightmapRoad HeightmapRoad::makeSinusoidal(int nx, int nz,
                                            double dx, double dz,
                                            double originX, double originZ,
                                            double amplitude, double wavelength,
                                            double mu) {
    std::vector<double> h(static_cast<size_t>(nx) * static_cast<size_t>(nz), 0.0);
    const double k = 2.0 * M_PI / std::max(wavelength, 1e-6);
    for (int iz = 0; iz < nz; ++iz) {
        double z = originZ + iz * dz;
        for (int ix = 0; ix < nx; ++ix) {
            double x = originX + ix * dx;
            h[static_cast<size_t>(iz) * nx + ix] =
                amplitude * std::sin(k * x) * std::cos(k * z);
        }
    }
    return HeightmapRoad(nx, nz, dx, dz, originX, originZ, std::move(h), mu);
}

double HeightmapRoad::sample(int ix, int iz) const {
    ix = std::max(0, std::min(nx_ - 1, ix));
    iz = std::max(0, std::min(nz_ - 1, iz));
    return h_[static_cast<size_t>(iz) * nx_ + ix];
}

double HeightmapRoad::height(double x, double z) const {
    double fx = (x - originX_) / dx_;
    double fz = (z - originZ_) / dz_;
    int ix = static_cast<int>(std::floor(fx));
    int iz = static_cast<int>(std::floor(fz));
    double tx = fx - ix;
    double tz = fz - iz;
    double h00 = sample(ix,     iz);
    double h10 = sample(ix + 1, iz);
    double h01 = sample(ix,     iz + 1);
    double h11 = sample(ix + 1, iz + 1);
    double h0 = h00 * (1.0 - tx) + h10 * tx;
    double h1 = h01 * (1.0 - tx) + h11 * tx;
    return h0 * (1.0 - tz) + h1 * tz;
}

Vec3 HeightmapRoad::normal(double x, double z) const {
    // Central finite-difference slopes.
    double hxp = height(x + dx_, z);
    double hxm = height(x - dx_, z);
    double hzp = height(x, z + dz_);
    double hzm = height(x, z - dz_);
    double dHdx = (hxp - hxm) / (2.0 * dx_);
    double dHdz = (hzp - hzm) / (2.0 * dz_);
    // Normal of surface y = h(x,z): (-dh/dx, 1, -dh/dz), normalized.
    return Vec3(-dHdx, 1.0, -dHdz).normalize();
}

} // namespace mb
