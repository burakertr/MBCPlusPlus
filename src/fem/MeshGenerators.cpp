#include "mb/fem/MeshGenerators.h"
#include <cmath>
#include <map>
#include <string>
#include <algorithm>  // std::clamp

namespace mb {

GmshMesh generateBoxTetMesh(double Lx, double Ly, double Lz,
                            int nx, int ny, int nz)
{
    GmshMesh mesh;
    double dx = Lx/nx, dy = Ly/ny, dz = Lz/nz;

    auto nodeIndex = [&](int ix, int iy, int iz) {
        return ix + (nx+1)*(iy + (ny+1)*iz);
    };

    int nid = 0;
    for (int iz = 0; iz <= nz; iz++)
        for (int iy = 0; iy <= ny; iy++)
            for (int ix = 0; ix <= nx; ix++)
                mesh.nodes.push_back({nid++, ix*dx, iy*dy, iz*dz});

    // Alternating 5-tet Kuhn triangulation: adjacent cells use
    // opposite body diagonals → face-compatible & symmetric mesh.
    int eid = 0;
    for (int iz = 0; iz < nz; iz++)
        for (int iy = 0; iy < ny; iy++)
            for (int ix = 0; ix < nx; ix++) {
                int v[8] = {
                    nodeIndex(ix,iy,iz), nodeIndex(ix+1,iy,iz),
                    nodeIndex(ix+1,iy+1,iz), nodeIndex(ix,iy+1,iz),
                    nodeIndex(ix,iy,iz+1), nodeIndex(ix+1,iy,iz+1),
                    nodeIndex(ix+1,iy+1,iz+1), nodeIndex(ix,iy+1,iz+1)
                };
                int tets[5][4];
                if ((ix + iy + iz) % 2 == 0) {
                    tets[0][0]=v[0]; tets[0][1]=v[1]; tets[0][2]=v[2]; tets[0][3]=v[5];
                    tets[1][0]=v[0]; tets[1][1]=v[2]; tets[1][2]=v[3]; tets[1][3]=v[7];
                    tets[2][0]=v[0]; tets[2][1]=v[4]; tets[2][2]=v[5]; tets[2][3]=v[7];
                    tets[3][0]=v[2]; tets[3][1]=v[5]; tets[3][2]=v[6]; tets[3][3]=v[7];
                    tets[4][0]=v[0]; tets[4][1]=v[2]; tets[4][2]=v[5]; tets[4][3]=v[7];
                } else {
                    tets[0][0]=v[0]; tets[0][1]=v[1]; tets[0][2]=v[3]; tets[0][3]=v[4];
                    tets[1][0]=v[1]; tets[1][1]=v[2]; tets[1][2]=v[3]; tets[1][3]=v[6];
                    tets[2][0]=v[1]; tets[2][1]=v[4]; tets[2][2]=v[5]; tets[2][3]=v[6];
                    tets[3][0]=v[3]; tets[3][1]=v[4]; tets[3][2]=v[6]; tets[3][3]=v[7];
                    tets[4][0]=v[1]; tets[4][1]=v[3]; tets[4][2]=v[4]; tets[4][3]=v[6];
                }
                for (auto& t : tets)
                    mesh.elements.push_back({eid++, 4, {t[0],t[1],t[2],t[3]}});
            }

    return mesh;
}

GmshMesh generateBoxHexMesh(double Lx, double Ly, double Lz,
                            int nx, int ny, int nz)
{
    GmshMesh mesh;
    double dx = Lx / nx, dy = Ly / ny, dz = Lz / nz;

    auto nodeIndex = [&](int ix, int iy, int iz) {
        return ix + (nx + 1) * (iy + (ny + 1) * iz);
    };

    int nid = 0;
    for (int iz = 0; iz <= nz; iz++)
        for (int iy = 0; iy <= ny; iy++)
            for (int ix = 0; ix <= nx; ix++)
                mesh.nodes.push_back({nid++, ix * dx, iy * dy, iz * dz});

    int eid = 0;
    for (int iz = 0; iz < nz; iz++)
        for (int iy = 0; iy < ny; iy++)
            for (int ix = 0; ix < nx; ix++) {
                int v0 = nodeIndex(ix,   iy,   iz);
                int v1 = nodeIndex(ix+1, iy,   iz);
                int v2 = nodeIndex(ix+1, iy+1, iz);
                int v3 = nodeIndex(ix,   iy+1, iz);
                int v4 = nodeIndex(ix,   iy,   iz+1);
                int v5 = nodeIndex(ix+1, iy,   iz+1);
                int v6 = nodeIndex(ix+1, iy+1, iz+1);
                int v7 = nodeIndex(ix,   iy+1, iz+1);
                mesh.elements.push_back({eid++, 5, {v0,v1,v2,v3,v4,v5,v6,v7}});
            }

    return mesh;
}

GmshMesh generateCylinderTetMesh(double R, double L,
                                  int nR, int nT, int nZ,
                                  double innerRadius)
{
    GmshMesh mesh;
    double dz = L / nZ;
    double Ri = std::clamp(innerRadius, 0.0, R * 0.999999);
    bool isSolid = Ri <= 1e-12;

    std::map<std::string, int> nodeMap;
    int nid = 0;

    auto getOrCreate = [&](int ir, int it, int iz) -> int {
        bool collapse = isSolid && ir == 0;
        std::string key = collapse
            ? "0-0-" + std::to_string(iz)
            : std::to_string(ir) + "-" + std::to_string(it % nT) + "-" + std::to_string(iz);

        auto it2 = nodeMap.find(key);
        if (it2 != nodeMap.end()) return it2->second;

        double x, y, z;
        if (collapse) { x = 0; y = 0; }
        else {
            double r = isSolid ? ((double)ir/nR)*R : Ri + ((double)ir/nR)*(R-Ri);
            double theta = 2.0*M_PI*(it % nT) / nT;
            x = r*std::cos(theta); y = r*std::sin(theta);
        }
        z = -L/2.0 + iz*dz;

        int id = nid++;
        nodeMap[key] = id;
        mesh.nodes.push_back({id, x, y, z});
        return id;
    };

    for (int iz = 0; iz <= nZ; iz++)
        for (int ir = 0; ir <= nR; ir++) {
            if (isSolid && ir == 0) getOrCreate(0, 0, iz);
            else for (int it = 0; it < nT; it++) getOrCreate(ir, it, iz);
        }

    int eid = 0;
    for (int iz = 0; iz < nZ; iz++)
        for (int ir = 0; ir < nR; ir++)
            for (int it = 0; it < nT; it++) {
                int it1 = (it+1) % nT;
                if (isSolid && ir == 0) {
                    int c0 = getOrCreate(0,0,iz),   c1 = getOrCreate(0,0,iz+1);
                    int a0 = getOrCreate(1,it,iz),   b0 = getOrCreate(1,it1,iz);
                    int a1 = getOrCreate(1,it,iz+1), b1 = getOrCreate(1,it1,iz+1);
                    mesh.elements.push_back({eid++, 4, {c0,a0,b0,c1}});
                    mesh.elements.push_back({eid++, 4, {a0,b0,c1,a1}});
                    mesh.elements.push_back({eid++, 4, {b0,c1,a1,b1}});
                } else {
                    int v[8] = {
                        getOrCreate(ir,  it, iz),  getOrCreate(ir+1,it, iz),
                        getOrCreate(ir+1,it1,iz),  getOrCreate(ir,  it1,iz),
                        getOrCreate(ir,  it, iz+1),getOrCreate(ir+1,it, iz+1),
                        getOrCreate(ir+1,it1,iz+1),getOrCreate(ir,  it1,iz+1)
                    };
                    int tets[6][4] = {
                        {v[0],v[1],v[2],v[6]}, {v[0],v[1],v[6],v[5]},
                        {v[0],v[4],v[5],v[6]}, {v[0],v[2],v[3],v[6]},
                        {v[0],v[3],v[7],v[6]}, {v[0],v[4],v[6],v[7]}
                    };
                    for (auto& t : tets)
                        mesh.elements.push_back({eid++, 4, {t[0],t[1],t[2],t[3]}});
                }
            }

    return mesh;
}

} // namespace mb
