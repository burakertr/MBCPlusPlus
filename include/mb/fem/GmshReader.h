#pragma once
#include "mb/fem/ANCFTypes.h"
#include <string>

namespace mb {

/**
 * Gmsh .msh file reader.
 * Supports Gmsh format v2.x and v4.x (ASCII).
 * Extracts nodes and 4-node tetrahedral elements (type 4).
 */
class GmshReader {
public:
    /// Parse a Gmsh .msh file content string
    static GmshMesh parse(const std::string& content);

private:
    static GmshMesh parseV2(const std::vector<std::string>& lines);
    static GmshMesh parseV4(const std::vector<std::string>& lines);
};

} // namespace mb
