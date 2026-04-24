#include "mb/fem/GmshReader.h"
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace mb {

static std::vector<std::string> splitLines(const std::string& content) {
    std::vector<std::string> lines;
    std::istringstream iss(content);
    std::string line;
    while (std::getline(iss, line)) {
        // Trim whitespace
        size_t start = line.find_first_not_of(" \t\r");
        size_t end = line.find_last_not_of(" \t\r");
        if (start == std::string::npos) lines.push_back("");
        else lines.push_back(line.substr(start, end - start + 1));
    }
    return lines;
}

static int findLine(const std::vector<std::string>& lines, const std::string& tag) {
    for (int i = 0; i < (int)lines.size(); i++)
        if (lines[i] == tag) return i;
    return -1;
}

GmshMesh GmshReader::parse(const std::string& content) {
    auto lines = splitLines(content);

    int fmtIdx = findLine(lines, "$MeshFormat");
    if (fmtIdx < 0) throw std::runtime_error("Not a valid Gmsh .msh file");

    std::istringstream vss(lines[fmtIdx + 1]);
    double version; vss >> version;

    if (version >= 4.0)
        return parseV4(lines);
    return parseV2(lines);
}

GmshMesh GmshReader::parseV2(const std::vector<std::string>& lines) {
    GmshMesh mesh;

    int i = findLine(lines, "$Nodes");
    if (i < 0) throw std::runtime_error("$Nodes section not found");

    int numNodes; { std::istringstream ss(lines[++i]); ss >> numNodes; }

    for (int n = 0; n < numNodes; n++) {
        ++i;
        std::istringstream ss(lines[i]);
        GmshNode nd;
        ss >> nd.id >> nd.x >> nd.y >> nd.z;
        mesh.nodes.push_back(nd);
    }

    i = findLine(lines, "$Elements");
    if (i < 0) throw std::runtime_error("$Elements section not found");

    int numElements; { std::istringstream ss(lines[++i]); ss >> numElements; }

    for (int n = 0; n < numElements; n++) {
        ++i;
        std::istringstream ss(lines[i]);
        int elId, elType, numTags;
        ss >> elId >> elType >> numTags;
        // Skip tags
        for (int t = 0; t < numTags; t++) { int dummy; ss >> dummy; }
        // Read node ids
        std::vector<int> nodeIds;
        int nid;
        while (ss >> nid) nodeIds.push_back(nid);

        if (elType == 4 || elType == 5) {
            mesh.elements.push_back({elId, elType, nodeIds});
        }
    }

    return mesh;
}

GmshMesh GmshReader::parseV4(const std::vector<std::string>& lines) {
    GmshMesh mesh;

    int i = findLine(lines, "$Nodes");
    if (i < 0) throw std::runtime_error("$Nodes section not found");
    i++;

    std::istringstream hss(lines[i]);
    int numBlocks, totalNodes;
    hss >> numBlocks >> totalNodes;
    i++;

    for (int b = 0; b < numBlocks; b++) {
        std::istringstream bss(lines[i]);
        int entityDim, entityTag, parametric, numNodesBlock;
        bss >> entityDim >> entityTag >> parametric >> numNodesBlock;
        i++;

        std::vector<int> tags;
        for (int n = 0; n < numNodesBlock; n++) {
            int tag; std::istringstream ts(lines[i]); ts >> tag;
            tags.push_back(tag);
            i++;
        }
        for (int n = 0; n < numNodesBlock; n++) {
            std::istringstream cs(lines[i]);
            GmshNode nd;
            nd.id = tags[n];
            cs >> nd.x >> nd.y >> nd.z;
            mesh.nodes.push_back(nd);
            i++;
        }
    }

    i = findLine(lines, "$Elements");
    if (i < 0) throw std::runtime_error("$Elements section not found");
    i++;

    std::istringstream ehss(lines[i]);
    int numElBlocks;
    ehss >> numElBlocks;
    i++;

    for (int b = 0; b < numElBlocks; b++) {
        std::istringstream bss(lines[i]);
        int edim, etag, elType, numElBlock;
        bss >> edim >> etag >> elType >> numElBlock;
        i++;

        for (int n = 0; n < numElBlock; n++) {
            std::istringstream ess(lines[i]);
            int elId;
            ess >> elId;
            std::vector<int> nodeIds;
            int nid;
            while (ess >> nid) nodeIds.push_back(nid);

            if (elType == 4 || elType == 5) {
                mesh.elements.push_back({elId, elType, nodeIds});
            }
            i++;
        }
    }

    return mesh;
}

} // namespace mb
