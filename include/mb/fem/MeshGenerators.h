#pragma once
#include "mb/fem/ANCFTypes.h"

namespace mb {

/// Generate a box-shaped tetrahedral mesh using alternating 5-tet Kuhn triangulation.
/// Nodes are laid out on an (nx+1)×(ny+1)×(nz+1) grid; each cell is split into 5 tets.
GmshMesh generateBoxTetMesh(double Lx, double Ly, double Lz,
                            int nx, int ny, int nz);

/// Generate a box-shaped hexahedral (hex8) mesh.
/// Each cell maps directly to one hex element in Gmsh type-5 format.
GmshMesh generateBoxHexMesh(double Lx, double Ly, double Lz,
                            int nx, int ny, int nz);

/// Generate a cylindrical tetrahedral mesh.
/// @param R          Outer radius
/// @param L          Length along Z axis (centred at z=0)
/// @param nR         Radial divisions
/// @param nT         Circumferential divisions
/// @param nZ         Axial divisions
/// @param innerRadius Optional inner bore radius (0 = solid cylinder)
GmshMesh generateCylinderTetMesh(double R, double L,
                                  int nR, int nT, int nZ,
                                  double innerRadius = 0);

} // namespace mb
