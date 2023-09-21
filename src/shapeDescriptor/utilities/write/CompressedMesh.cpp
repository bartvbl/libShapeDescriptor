#include "CompressedMesh.h"
#include <cstdint>
#include <vector>
#include <shapeDescriptor/utilities/fileutils.h>

void ShapeDescriptor::utilities::writeCompressedMesh(const ShapeDescriptor::cpu::Mesh &mesh, const std::filesystem::path &filePath) {
    bool containsNormals = mesh.normals != nullptr;
    bool containsVertexColours = mesh.vertexColours != nullptr;

    const unsigned int headerSize = 2 * sizeof(uint64_t) + sizeof(uint32_t);
    const size_t vertexSize = mesh.vertexCount * sizeof(ShapeDescriptor::cpu::float3);
    const size_t normalSize = mesh.vertexCount * sizeof(ShapeDescriptor::cpu::float3) * (containsNormals ? 1 : 0);
    const size_t colourSize = mesh.vertexCount * sizeof(ShapeDescriptor::cpu::uchar4) * (containsVertexColours ? 1 : 0);
    std::vector<uint8_t> fileBuffer(headerSize + vertexSize + normalSize + colourSize);
    uint8_t* bufferPointer = fileBuffer.data();

    const uint64_t magic = 0x4C53532D4D455348;
    *reinterpret_cast<uint64_t*>(bufferPointer) = magic;
    bufferPointer += sizeof(uint64_t);

    const uint64_t vertexCount = mesh.vertexCount;
    *reinterpret_cast<uint64_t*>(bufferPointer) = vertexCount;
    bufferPointer += sizeof(uint64_t);

    const uint32_t flagContainsNormals = containsNormals ? 1 : 0;
    const uint32_t flagContainsVertexColours = containsVertexColours ? 2 : 0;
    const uint32_t flags = flagContainsNormals | flagContainsVertexColours;
    *reinterpret_cast<uint32_t*>(bufferPointer) = flags;
    bufferPointer += sizeof(uint32_t);

    std::copy(mesh.vertices, mesh.vertices + mesh.vertexCount, reinterpret_cast<ShapeDescriptor::cpu::float3*>(bufferPointer));
    bufferPointer += mesh.vertexCount * sizeof(ShapeDescriptor::cpu::float3);

    if(containsNormals) {
        std::copy(mesh.normals, mesh.normals + mesh.vertexCount, reinterpret_cast<ShapeDescriptor::cpu::float3*>(bufferPointer));
        bufferPointer += mesh.vertexCount * sizeof(ShapeDescriptor::cpu::float3);
    }

    if(containsVertexColours) {
        std::copy(mesh.vertexColours, mesh.vertexColours + mesh.vertexCount, reinterpret_cast<ShapeDescriptor::cpu::uchar4*>(bufferPointer));
        bufferPointer += mesh.vertexCount * sizeof(ShapeDescriptor::cpu::uchar4);
    }

    assert(bufferPointer == fileBuffer.size());

    ShapeDescriptor::utilities::writeCompressedFile((char*) fileBuffer.data(), fileBuffer.size(), filePath, 4);
}
