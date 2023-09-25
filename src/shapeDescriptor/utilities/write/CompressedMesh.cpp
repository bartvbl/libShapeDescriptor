#include "CompressedMesh.h"
#include <cstdint>
#include <vector>
#include <shapeDescriptor/utilities/fileutils.h>
#include <unordered_set>
#include <unordered_map>
#include <meshoptimizer.h>

template<typename T> uint8_t* write(const T& data, uint8_t* bufferPointer) {
    *reinterpret_cast<T*>(bufferPointer) = data;
    bufferPointer += sizeof(T);
    return bufferPointer;
}

void ShapeDescriptor::utilities::writeCompressedMesh(const ShapeDescriptor::cpu::Mesh &mesh, const std::filesystem::path &filePath) {
    // limits supported number of triangles per triangle strip to 2B
    const uint32_t TRIANGLE_STRIP_END_FLAG = 0x1U << 31;

    meshopt_encodeVertexVersion(0);
    meshopt_encodeIndexVersion(1);

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


    // -- Compressing vertex positions --

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

    size_t vertexBufferSizeBound = meshopt_encodeVertexBufferBound(condensedVertices.size(), sizeof(ShapeDescriptor::cpu::float3));
    std::vector<unsigned char> compressedVertexBuffer(vertexBufferSizeBound);
    size_t compressedVertexBufferSize = meshopt_encodeVertexBuffer(compressedVertexBuffer.data(), compressedVertexBuffer.size(), condensedVertices.data(), condensedVertices.size(), sizeof(ShapeDescriptor::cpu::float3));
    compressedVertexBuffer.resize(compressedVertexBufferSize);

    size_t indexBufferSizeBound = meshopt_encodeIndexBufferBound(vertexIndexBuffer.size(), mesh.vertexCount);
    std::vector<unsigned char> compressedIndexBuffer(indexBufferSizeBound);
    size_t compressedIndexBufferSize = meshopt_encodeIndexBuffer(compressedIndexBuffer.data(), compressedIndexBuffer.size(), vertexIndexBuffer.data(), vertexIndexBuffer.size());
    compressedIndexBuffer.resize(compressedIndexBufferSize);


    // -- Compressing normals --

    std::vector<unsigned char> compressedNormalBuffer;
    size_t compressedNormalBufferSize = 0;

    std::vector<unsigned char> compressedNormalIndexBuffer;
    size_t compressedNormalIndexBufferSize = 0;

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

        size_t normalBufferSizeBound = meshopt_encodeVertexBufferBound(condensedNormals.size(), sizeof(ShapeDescriptor::cpu::float3));
        compressedNormalBuffer.resize(normalBufferSizeBound);
        compressedNormalBufferSize = meshopt_encodeVertexBuffer(compressedNormalBuffer.data(), compressedNormalBuffer.size(), condensedNormals.data(), condensedNormals.size(), sizeof(ShapeDescriptor::cpu::float3));
        compressedNormalBuffer.resize(compressedNormalBufferSize);

        size_t normalIndexBufferSizeBound = meshopt_encodeIndexBufferBound(normalIndexBuffer.size(), mesh.vertexCount);
        compressedNormalIndexBuffer.resize(normalIndexBufferSizeBound);
        compressedNormalIndexBufferSize = meshopt_encodeIndexBuffer(compressedNormalIndexBuffer.data(), compressedNormalIndexBuffer.size(), normalIndexBuffer.data(), normalIndexBuffer.size());
        compressedNormalIndexBuffer.resize(compressedNormalIndexBufferSize);
    }

    size_t compressedColourBufferSize = 0;

    if(containsVertexColours) {
        /* ... */
    }


    // Header consists of:
    // - 8 byte magic
    // - 4 byte file version
    // - 4 byte flags
    // - 4 byte uncondensed/original vertex count
    // - 2 x 4 byte condensed vertex/normal buffers lengths
    // - 4 x 8 byte compressed vertex/normal/vertex_index/normal_index buffer sizes in bytes
    // - 8 byte compressed vertex colour buffer
    // (note: 32-bit colour is the same size as the index buffer, and therefore does not save on file size. It thus does not have an index buffer)
    const uint32_t headerSize = 6 * sizeof(uint64_t) + 5 * sizeof(uint32_t);
    const size_t vertexSize = compressedVertexBufferSize;
    const size_t normalSize = compressedNormalBufferSize;
    const size_t colourSize = mesh.vertexCount * sizeof(ShapeDescriptor::cpu::uchar4) * (containsVertexColours ? 1 : 0);
    const size_t vertexIndexSize = compressedIndexBufferSize;
    const size_t normalIndexSize = compressedNormalIndexBufferSize;
    std::vector<uint8_t> fileBuffer(headerSize + vertexSize + normalSize + colourSize + vertexIndexSize + normalIndexSize);
    uint8_t* bufferPointer = fileBuffer.data();



    // header: magic
    const uint64_t magic = 0x4C53532D4D455348;
    bufferPointer = write(magic, bufferPointer);

    // header: version
    const uint32_t fileSpecVersion = 1;
    bufferPointer = write(fileSpecVersion, bufferPointer);

    // header: flags
    const uint32_t flagContainsNormals = containsNormals ? 1 : 0;
    const uint32_t flagContainsVertexColours = containsVertexColours ? 2 : 0;
    const uint32_t flags = flagContainsNormals | flagContainsVertexColours;
    bufferPointer = write(flags, bufferPointer);

    // header: uncondensed vertex count
    const uint32_t vertexCount = mesh.vertexCount; // cast to 32 bit
    bufferPointer = write(vertexCount, bufferPointer);

    // header: condensed buffer lengths
    const uint32_t condensedVertexCount = condensedVertices.size();
    const uint32_t condensedNormalCount = condensedNormals.size();
    bufferPointer = write(condensedVertexCount, bufferPointer);
    bufferPointer = write(condensedNormalCount, bufferPointer);

    // header: compressed vertex/normal/vertex_index/vertex_normal buffer sizes
    bufferPointer = write(compressedVertexBufferSize, bufferPointer);
    bufferPointer = write(compressedIndexBufferSize, bufferPointer);
    bufferPointer = write(compressedNormalBufferSize, bufferPointer);
    bufferPointer = write(compressedNormalIndexBufferSize, bufferPointer);
    bufferPointer = write(compressedColourBufferSize, bufferPointer);

    // contents: vertex data
    std::copy(compressedVertexBuffer.begin(), compressedVertexBuffer.end(), bufferPointer);
    bufferPointer += compressedVertexBufferSize;
    std::copy(compressedIndexBuffer.begin(), compressedIndexBuffer.end(), bufferPointer);
    bufferPointer += compressedIndexBufferSize;

    // contents: normal data
    if(containsNormals) {
        std::copy(compressedNormalBuffer.begin(), compressedNormalBuffer.end(), bufferPointer);
        bufferPointer += compressedNormalBufferSize;
        std::copy(compressedNormalIndexBuffer.begin(), compressedNormalIndexBuffer.end(), bufferPointer);
        bufferPointer += compressedNormalIndexBufferSize;
    }

    // contents: colour data
    if(containsVertexColours) {
        std::copy(mesh.vertexColours, mesh.vertexColours + mesh.vertexCount, reinterpret_cast<ShapeDescriptor::cpu::uchar4*>(bufferPointer));
        bufferPointer += mesh.vertexCount * sizeof(ShapeDescriptor::cpu::uchar4);
    }

    assert(bufferPointer == fileBuffer.data() + fileBuffer.size());

    ShapeDescriptor::utilities::writeCompressedFile((char*) fileBuffer.data(), fileBuffer.size(), filePath, 4);
}
