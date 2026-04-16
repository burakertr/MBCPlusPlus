#include "mb/fem/FlexibleContactDetector.h"
#include <cmath>
#include <algorithm>
#include <map>
#include <set>

namespace mb {

// ─── Surface Extraction ──────────────────────────────────────

std::vector<SurfaceTriangle> extractSurfaceTriangles(FlexibleBody& body) {
    std::map<std::string, std::pair<SurfaceTriangle, int>> faceCount;

    auto makeKey = [](int a, int b, int c) {
        int arr[3] = {a, b, c};
        std::sort(arr, arr + 3);
        return std::to_string(arr[0]) + "," + std::to_string(arr[1]) + "," + std::to_string(arr[2]);
    };

    for (const auto& elem : body.elements) {
        auto& nid = elem.nodeIds;
        int faces[4][3] = {
            {nid[0], nid[1], nid[2]},
            {nid[0], nid[1], nid[3]},
            {nid[0], nid[2], nid[3]},
            {nid[1], nid[2], nid[3]}
        };
        for (auto& face : faces) {
            auto key = makeKey(face[0], face[1], face[2]);
            auto it = faceCount.find(key);
            if (it != faceCount.end()) {
                it->second.second++;
            } else {
                faceCount[key] = {{face[0], face[1], face[2]}, 1};
            }
        }
    }

    std::vector<SurfaceTriangle> surface;
    for (auto& [key, val] : faceCount) {
        if (val.second == 1)
            surface.push_back(val.first);
    }

    // Orient normals outward (away from centroid)
    int N = (int)body.nodes.size();
    if (N > 0) {
        double cx = 0, cy = 0, cz = 0;
        for (const auto& nd : body.nodes) {
            cx += nd.q[0]; cy += nd.q[1]; cz += nd.q[2];
        }
        cx /= N; cy /= N; cz /= N;

        for (auto& tri : surface) {
            auto& p0 = body.nodes[tri.n0].q;
            auto& p1 = body.nodes[tri.n1].q;
            auto& p2 = body.nodes[tri.n2].q;

            double fx = (p0[0]+p1[0]+p2[0])/3;
            double fy = (p0[1]+p1[1]+p2[1])/3;
            double fz = (p0[2]+p1[2]+p2[2])/3;

            double e1x = p1[0]-p0[0], e1y = p1[1]-p0[1], e1z = p1[2]-p0[2];
            double e2x = p2[0]-p0[0], e2y = p2[1]-p0[1], e2z = p2[2]-p0[2];
            double nx = e1y*e2z - e1z*e2y;
            double ny = e1z*e2x - e1x*e2z;
            double nz = e1x*e2y - e1y*e2x;

            double dx = fx-cx, dy = fy-cy, dz = fz-cz;
            if (nx*dx + ny*dy + nz*dz < 0)
                std::swap(tri.n1, tri.n2);
        }
    }

    return surface;
}

std::set<int> extractSurfaceNodeIndices(const std::vector<SurfaceTriangle>& tris) {
    std::set<int> s;
    for (const auto& t : tris) {
        s.insert(t.n0); s.insert(t.n1); s.insert(t.n2);
    }
    return s;
}

// ─── AABB Helpers ────────────────────────────────────────────

AABB computeBodyAABB(const FlexibleBody& body, const std::set<int>& nodeIndices) {
    AABB aabb{1e30, 1e30, 1e30, -1e30, -1e30, -1e30};
    for (int ni : nodeIndices) {
        double x = body.nodes[ni].q[0], y = body.nodes[ni].q[1], z = body.nodes[ni].q[2];
        if (x < aabb.minX) aabb.minX = x; if (x > aabb.maxX) aabb.maxX = x;
        if (y < aabb.minY) aabb.minY = y; if (y > aabb.maxY) aabb.maxY = y;
        if (z < aabb.minZ) aabb.minZ = z; if (z > aabb.maxZ) aabb.maxZ = z;
    }
    return aabb;
}

bool aabbOverlap(const AABB& a, const AABB& b, double margin) {
    return a.minX - margin <= b.maxX && a.maxX + margin >= b.minX &&
           a.minY - margin <= b.maxY && a.maxY + margin >= b.minY &&
           a.minZ - margin <= b.maxZ && a.maxZ + margin >= b.minZ;
}

// ─── Closest Point on Triangle ───────────────────────────────

static Vec3 closestPointOnTri(const Vec3& P, const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 ab = v1 - v0, ac = v2 - v0, ap = P - v0;
    double d1 = ab.dot(ap), d2 = ac.dot(ap);
    if (d1 <= 0 && d2 <= 0) return v0;

    Vec3 bp = P - v1;
    double d3 = ab.dot(bp), d4 = ac.dot(bp);
    if (d3 >= 0 && d4 <= d3) return v1;

    double vc = d1*d4 - d3*d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        double v = d1 / (d1 - d3);
        return v0 + ab * v;
    }

    Vec3 cp = P - v2;
    double d5 = ab.dot(cp), d6 = ac.dot(cp);
    if (d6 >= 0 && d5 <= d6) return v2;

    double vb = d5*d2 - d1*d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        double w = d2 / (d2 - d6);
        return v0 + ac * w;
    }

    double va = d3*d6 - d5*d4;
    if (va <= 0 && (d4-d3) >= 0 && (d5-d6) >= 0) {
        double w = (d4-d3) / ((d4-d3) + (d5-d6));
        return v1 + (v2 - v1) * w;
    }

    double denom = 1.0 / (va + vb + vc);
    double v = vb * denom, w = vc * denom;
    return v0 + ab * v + ac * w;
}

// ─── Node-to-Surface Detection ───────────────────────────────

std::vector<FlexContact> FlexibleContactDetector::detectNodeToSurface(
    FlexibleBody& nodeBody,
    const std::set<int>& nodeIndices,
    FlexibleBody& surfBody,
    const std::vector<SurfaceTriangle>& surfTris)
{
    std::vector<FlexContact> contacts;
    int nTris = (int)surfTris.size();
    if (nTris == 0) return contacts;
    double margin = contactMargin;

    // Pre-compute per-triangle plane data
    int triDataLen = nTris * 7;
    if ((int)triPlaneData_.size() < triDataLen)
        triPlaneData_.resize(triDataLen);

    double sMinX = 1e30, sMinY = 1e30, sMinZ = 1e30;
    double sMaxX = -1e30, sMaxY = -1e30, sMaxZ = -1e30;

    for (int ti = 0; ti < nTris; ti++) {
        const auto& tri = surfTris[ti];
        auto& q0 = surfBody.nodes[tri.n0].q;
        auto& q1 = surfBody.nodes[tri.n1].q;
        auto& q2 = surfBody.nodes[tri.n2].q;

        double v0x = q0[0], v0y = q0[1], v0z = q0[2];
        double v1x = q1[0], v1y = q1[1], v1z = q1[2];
        double v2x = q2[0], v2y = q2[1], v2z = q2[2];

        // Track AABB
        sMinX = std::min({sMinX, v0x, v1x, v2x}); sMaxX = std::max({sMaxX, v0x, v1x, v2x});
        sMinY = std::min({sMinY, v0y, v1y, v2y}); sMaxY = std::max({sMaxY, v0y, v1y, v2y});
        sMinZ = std::min({sMinZ, v0z, v1z, v2z}); sMaxZ = std::max({sMaxZ, v0z, v1z, v2z});

        double e1x = v1x-v0x, e1y = v1y-v0y, e1z = v1z-v0z;
        double e2x = v2x-v0x, e2y = v2y-v0y, e2z = v2z-v0z;
        double nx = e1y*e2z - e1z*e2y;
        double ny = e1z*e2x - e1x*e2z;
        double nz = e1x*e2y - e1y*e2x;
        double len = std::sqrt(nx*nx + ny*ny + nz*nz);

        int off = ti * 7;
        triPlaneData_[off] = v0x; triPlaneData_[off+1] = v0y; triPlaneData_[off+2] = v0z;
        triPlaneData_[off+3] = nx; triPlaneData_[off+4] = ny; triPlaneData_[off+5] = nz;
        triPlaneData_[off+6] = len > 1e-20 ? 1.0/len : 0;
    }

    double inflate = margin + maxDepth;
    sMinX -= inflate; sMinY -= inflate; sMinZ -= inflate;
    sMaxX += inflate; sMaxY += inflate; sMaxZ += inflate;

    for (int ni : nodeIndices) {
        auto& qn = nodeBody.nodes[ni].q;
        double px = qn[0], py = qn[1], pz = qn[2];

        if (px < sMinX || px > sMaxX || py < sMinY || py > sMaxY || pz < sMinZ || pz > sMaxZ)
            continue;

        double maxSD = -1e30;
        int maxSDFace = -1;

        for (int ti = 0; ti < nTris; ti++) {
            int off = ti * 7;
            double invLen = triPlaneData_[off+6];
            if (invLen == 0) continue;
            double dx = px - triPlaneData_[off];
            double dy = py - triPlaneData_[off+1];
            double dz = pz - triPlaneData_[off+2];
            double sd = (dx*triPlaneData_[off+3] + dy*triPlaneData_[off+4] + dz*triPlaneData_[off+5]) * invLen;
            if (sd > maxSD) { maxSD = sd; maxSDFace = ti; }
        }

        if (maxSDFace < 0 || maxSD > margin) continue;

        double depth = margin - maxSD;
        if (depth < minDepth || depth > maxDepth) continue;

        const auto& tri = surfTris[maxSDFace];
        Vec3 v0(surfBody.nodes[tri.n0].q[0], surfBody.nodes[tri.n0].q[1], surfBody.nodes[tri.n0].q[2]);
        Vec3 v1(surfBody.nodes[tri.n1].q[0], surfBody.nodes[tri.n1].q[1], surfBody.nodes[tri.n1].q[2]);
        Vec3 v2(surfBody.nodes[tri.n2].q[0], surfBody.nodes[tri.n2].q[1], surfBody.nodes[tri.n2].q[2]);

        Vec3 e1 = v1 - v0, e2 = v2 - v0;
        Vec3 rawN = e1.cross(e2);
        double nLen = rawN.length();
        if (nLen < 1e-20) continue;
        Vec3 n = rawN * (1.0/nLen);

        Vec3 P(px, py, pz);
        Vec3 closest = closestPointOnTri(P, v0, v1, v2);

        contacts.push_back({&nodeBody, ni, &surfBody, maxSDFace, closest, n, depth});
    }

    return contacts;
}

// ─── Node-to-Ground Detection ────────────────────────────────

std::vector<FlexContact> FlexibleContactDetector::detectNodeToGround(
    FlexibleBody& body,
    const std::set<int>& nodeIndices,
    const GroundPlane& ground)
{
    std::vector<FlexContact> contacts;
    double gnx = ground.normal.x, gny = ground.normal.y, gnz = ground.normal.z;

    for (int ni : nodeIndices) {
        auto& q = body.nodes[ni].q;
        double px = q[0], py = q[1], pz = q[2];
        double signedDist = (px*gnx + py*gny + pz*gnz) - ground.y;
        if (signedDist >= 0) continue;

        double depth = -signedDist;
        if (depth < minDepth) continue;

        Vec3 point(px - gnx*signedDist, py - gny*signedDist, pz - gnz*signedDist);
        contacts.push_back({&body, ni, nullptr, -1, point, ground.normal, std::min(depth, maxDepth)});
    }

    return contacts;
}

// ─── SAT Detection ───────────────────────────────────────────

std::vector<FlexContact> FlexibleContactDetector::detectSAT(
    FlexibleBody& bodyA,
    const std::vector<SurfaceTriangle>& trisA,
    FlexibleBody& bodyB,
    const std::vector<SurfaceTriangle>& trisB)
{
    int nA = (int)bodyA.nodes.size(), nB = (int)bodyB.nodes.size();
    std::vector<double> posA(nA*3), posB(nB*3);
    for (int i = 0; i < nA; i++) {
        posA[i*3] = bodyA.nodes[i].q[0]; posA[i*3+1] = bodyA.nodes[i].q[1]; posA[i*3+2] = bodyA.nodes[i].q[2];
    }
    for (int i = 0; i < nB; i++) {
        posB[i*3] = bodyB.nodes[i].q[0]; posB[i*3+1] = bodyB.nodes[i].q[1]; posB[i*3+2] = bodyB.nodes[i].q[2];
    }

    // Collect SAT axes
    std::vector<double> axes;
    auto addNormals = [&](const std::vector<SurfaceTriangle>& tris, const std::vector<double>& pos) {
        for (const auto& tri : tris) {
            int o0 = tri.n0*3, o1 = tri.n1*3, o2 = tri.n2*3;
            double e1x = pos[o1]-pos[o0], e1y = pos[o1+1]-pos[o0+1], e1z = pos[o1+2]-pos[o0+2];
            double e2x = pos[o2]-pos[o0], e2y = pos[o2+1]-pos[o0+1], e2z = pos[o2+2]-pos[o0+2];
            double nx = e1y*e2z-e1z*e2y, ny = e1z*e2x-e1x*e2z, nz = e1x*e2y-e1y*e2x;
            double len2 = nx*nx + ny*ny + nz*nz;
            if (len2 > 1e-20) {
                double inv = 1.0/std::sqrt(len2);
                axes.push_back(nx*inv); axes.push_back(ny*inv); axes.push_back(nz*inv);
            }
        }
    };
    addNormals(trisA, posA);
    addNormals(trisB, posB);

    // Coordinate axes
    axes.push_back(1); axes.push_back(0); axes.push_back(0);
    axes.push_back(0); axes.push_back(1); axes.push_back(0);
    axes.push_back(0); axes.push_back(0); axes.push_back(1);

    int numAxes = (int)axes.size() / 3;
    double minOverlap = 1e30;
    double minAx = 0, minAy = 1, minAz = 0;

    for (int ai = 0; ai < numAxes; ai++) {
        double ax = axes[ai*3], ay = axes[ai*3+1], az = axes[ai*3+2];
        double minAProj = 1e30, maxAProj = -1e30;
        for (int i = 0; i < nA; i++) {
            double d = posA[i*3]*ax + posA[i*3+1]*ay + posA[i*3+2]*az;
            if (d < minAProj) minAProj = d; if (d > maxAProj) maxAProj = d;
        }
        double minBProj = 1e30, maxBProj = -1e30;
        for (int i = 0; i < nB; i++) {
            double d = posB[i*3]*ax + posB[i*3+1]*ay + posB[i*3+2]*az;
            if (d < minBProj) minBProj = d; if (d > maxBProj) maxBProj = d;
        }
        double overlap = std::min(maxAProj, maxBProj) - std::max(minAProj, minBProj);
        if (overlap <= 0) return {};
        if (overlap < minOverlap) {
            minOverlap = overlap; minAx = ax; minAy = ay; minAz = az;
        }
    }

    // Ensure normal from B to A
    double cAx = 0, cAy = 0, cAz = 0;
    for (int i = 0; i < nA; i++) { cAx += posA[i*3]; cAy += posA[i*3+1]; cAz += posA[i*3+2]; }
    cAx /= nA; cAy /= nA; cAz /= nA;
    double cBx = 0, cBy = 0, cBz = 0;
    for (int i = 0; i < nB; i++) { cBx += posB[i*3]; cBy += posB[i*3+1]; cBz += posB[i*3+2]; }
    cBx /= nB; cBy /= nB; cBz /= nB;

    if (minAx*(cAx-cBx) + minAy*(cAy-cBy) + minAz*(cAz-cBz) < 0) {
        minAx = -minAx; minAy = -minAy; minAz = -minAz;
    }

    double depth = std::min(minOverlap, maxDepth);
    if (depth < minDepth) return {};

    Vec3 minAxis(minAx, minAy, minAz);
    std::vector<FlexContact> contacts;

    // Deepest vertex of A into B
    {
        int bestIdx = 0; double bestProj = 1e30;
        for (int i = 0; i < nA; i++) {
            double p = posA[i*3]*minAx + posA[i*3+1]*minAy + posA[i*3+2]*minAz;
            if (p < bestProj) { bestProj = p; bestIdx = i; }
        }
        Vec3 pt(posA[bestIdx*3] - minAx*depth*0.5, posA[bestIdx*3+1] - minAy*depth*0.5, posA[bestIdx*3+2] - minAz*depth*0.5);
        contacts.push_back({&bodyA, bestIdx, &bodyB, 0, pt, minAxis, depth});
    }
    // Deepest vertex of B into A
    {
        double rnx = -minAx, rny = -minAy, rnz = -minAz;
        int bestIdx = 0; double bestProj = 1e30;
        for (int i = 0; i < nB; i++) {
            double p = posB[i*3]*rnx + posB[i*3+1]*rny + posB[i*3+2]*rnz;
            if (p < bestProj) { bestProj = p; bestIdx = i; }
        }
        Vec3 pt(posB[bestIdx*3] - rnx*depth*0.5, posB[bestIdx*3+1] - rny*depth*0.5, posB[bestIdx*3+2] - rnz*depth*0.5);
        contacts.push_back({&bodyB, bestIdx, &bodyA, 0, pt, Vec3(rnx, rny, rnz), depth});
    }

    return contacts;
}

} // namespace mb
