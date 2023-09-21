#include "CompressedMesh.h"
#include <cstdint>
#include <vector>
#include <shapeDescriptor/utilities/fileutils.h>
#include <unordered_set>
#include <unordered_map>

void ShapeDescriptor::utilities::writeCompressedMesh(const ShapeDescriptor::cpu::Mesh &mesh, const std::filesystem::path &filePath) {
    bool containsNormals = mesh.normals != nullptr;
    bool containsVertexColours = mesh.vertexColours != nullptr;

    std::vector<ShapeDescriptor::cpu::float3> condensedVertices;
    std::vector<uint32_t> vertexIndexBuffer(mesh.vertexCount);
    std::unordered_set<ShapeDescriptor::cpu::float3> seenUniqueVertices;
    std::unordered_map<ShapeDescriptor::cpu::float3, uint32_t> seenVerticesIndex;

    std::vector<ShapeDescriptor::cpu::float3> condensedNormals;
    std::vector<uint32_t> normalIndexBuffer;
    std::unordered_set<ShapeDescriptor::cpu::float3> seenUniqueNormals;
    std::unordered_map<ShapeDescriptor::cpu::float3, uint32_t> seenNormalsIndex;

    condensedVertices.reserve(mesh.vertexCount);
    if(containsNormals) {
        condensedNormals.reserve(mesh.vertexCount);
        normalIndexBuffer.resize(mesh.vertexCount);
    }



    for(uint32_t i = 0; i < mesh.vertexCount; i++) {
        const ShapeDescriptor::cpu::float3 vertex = mesh.vertices[i];
        if(seenUniqueVertices.find(vertex) == seenUniqueVertices.end()) {
            // Vertex has not been seen before
            seenUniqueVertices.insert(vertex);
            seenVerticesIndex[vertex] = condensedVertices.size();
            condensedVertices.push_back(vertex);
        }
        vertexIndexBuffer.at(i) = seenVerticesIndex.at(vertex);
    }

    if(containsNormals) {
        for(uint32_t i = 0; i < mesh.vertexCount; i++) {
            const ShapeDescriptor::cpu::float3 normal = mesh.normals[i];
            if(seenUniqueNormals.find(normal) == seenUniqueNormals.end()) {
                // Normal has not been seen before
                seenUniqueNormals.insert(normal);
                seenNormalsIndex[normal] = condensedNormals.size();
                condensedNormals.push_back(normal);
            }
            normalIndexBuffer.at(i) = seenNormalsIndex.at(normal);
        }
    }


    // Header consists of:
    // - 8 byte magic
    // - 4 byte flags
    // - 4 byte uncondensed/original vertex count
    // - 2 x 4 byte condensed vertex/normal buffers lengths
    // (note: 32-bit colour is the same size as the index buffer, and therefore does not save on file size. It is thus left uncondensed)
    const uint32_t headerSize = sizeof(uint64_t) + 4 * sizeof(uint32_t);
    const size_t vertexSize = condensedVertices.size() * sizeof(ShapeDescriptor::cpu::float3);
    const size_t normalSize = condensedNormals.size() * sizeof(ShapeDescriptor::cpu::float3);
    const size_t colourSize = mesh.vertexCount * sizeof(ShapeDescriptor::cpu::uchar4) * (containsVertexColours ? 1 : 0);
    const size_t vertexIndexSize = vertexIndexBuffer.size() * sizeof(uint32_t);
    const size_t normalIndexSize = normalIndexBuffer.size() * sizeof(uint32_t);
    std::vector<uint8_t> fileBuffer(headerSize + vertexSize + normalSize + colourSize + vertexIndexSize + normalIndexSize);
    uint8_t* bufferPointer = fileBuffer.data();

    // header: magic
    const uint64_t magic = 0x4C53532D4D455348;
    *reinterpret_cast<uint64_t*>(bufferPointer) = magic;
    bufferPointer += sizeof(uint64_t);

    // header: flags
    const uint32_t flagContainsNormals = containsNormals ? 1 : 0;
    const uint32_t flagContainsVertexColours = containsVertexColours ? 2 : 0;
    const uint32_t flags = flagContainsNormals | flagContainsVertexColours;
    *reinterpret_cast<uint32_t*>(bufferPointer) = flags;
    bufferPointer += sizeof(uint32_t);

    // header: uncondensed vertex count
    const uint32_t vertexCount = mesh.vertexCount;
    *reinterpret_cast<uint32_t*>(bufferPointer) = vertexCount;
    bufferPointer += sizeof(uint32_t);

    // header: condensed buffer lengths
    const uint32_t condensedVertexCount = condensedVertices.size();
    *reinterpret_cast<uint32_t*>(bufferPointer) = condensedVertexCount;
    bufferPointer += sizeof(uint32_t);

    const uint32_t condensedNormalCount = condensedNormals.size();
    *reinterpret_cast<uint32_t*>(bufferPointer) = condensedNormalCount;
    bufferPointer += sizeof(uint32_t);

    std::copy(condensedVertices.begin(), condensedVertices.end(), reinterpret_cast<ShapeDescriptor::cpu::float3*>(bufferPointer));
    bufferPointer += condensedVertices.size() * sizeof(ShapeDescriptor::cpu::float3);

    std::copy(vertexIndexBuffer.begin(), vertexIndexBuffer.end(), reinterpret_cast<uint32_t*>(bufferPointer));
    bufferPointer += vertexIndexBuffer.size() * sizeof(uint32_t);

    if(containsNormals) {
        std::copy(condensedNormals.begin(), condensedNormals.end(), reinterpret_cast<ShapeDescriptor::cpu::float3*>(bufferPointer));
        bufferPointer += condensedNormals.size() * sizeof(ShapeDescriptor::cpu::float3);

        std::copy(normalIndexBuffer.begin(), normalIndexBuffer.end(), reinterpret_cast<uint32_t*>(bufferPointer));
        bufferPointer += normalIndexBuffer.size() * sizeof(uint32_t);
    }

    if(containsVertexColours) {
        std::copy(mesh.vertexColours, mesh.vertexColours + mesh.vertexCount, reinterpret_cast<ShapeDescriptor::cpu::uchar4*>(bufferPointer));
        bufferPointer += mesh.vertexCount * sizeof(ShapeDescriptor::cpu::uchar4);
    }

    assert(bufferPointer == fileBuffer.size());

    ShapeDescriptor::utilities::writeCompressedFile((char*) fileBuffer.data(), fileBuffer.size(), filePath, 4);
}
