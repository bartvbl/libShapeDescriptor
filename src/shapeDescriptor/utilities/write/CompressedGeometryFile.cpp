#include "CompressedGeometryFile.h"
#include <cstdint>
#include <vector>
#include <shapeDescriptor/utilities/fileutils.h>
#include <unordered_set>
#include <unordered_map>
#include <meshoptimizer.h>
#include <iostream>
#include <shapeDescriptor/utilities/read/MeshLoadUtils.h>
#include <map>

struct Vertex {
    ShapeDescriptor::cpu::float3 position = {0, 0, 0};
    ShapeDescriptor::cpu::float3 normal = {0, 0, 0};
    ShapeDescriptor::cpu::uchar4 colour = {0, 0, 0, 255};

    bool operator==(const Vertex& other) const {
        return position == other.position && normal == other.normal && colour == other.colour;
    }
};

// Allow inclusion into std::set
namespace std {
    template <> struct hash<Vertex>
    {
        size_t operator()(const Vertex& p) const
        {
            return std::hash<ShapeDescriptor::cpu::float3>()(p.position)
                    ^ std::hash<ShapeDescriptor::cpu::float3>()(p.normal)
                    ^ std::hash<ShapeDescriptor::cpu::uchar4>()(p.colour);
        }
    };
}

template<typename T> uint8_t* write(const T& data, uint8_t* bufferPointer) {
    *reinterpret_cast<T*>(bufferPointer) = data;
    bufferPointer += sizeof(T);
    return bufferPointer;
}

uint8_t* write(std::vector<uint8_t>& data, uint8_t* bufferPointer) {
    std::copy(data.begin(), data.end(), bufferPointer);
    bufferPointer += data.size();
    return bufferPointer;
}

bool canMeshNormalsBeComputedExactly(const ShapeDescriptor::cpu::float3* vertices,
                                     const ShapeDescriptor::cpu::float3* normals,
                                     size_t vertexCount) {
    bool normalsEquivalent = true;
    for(size_t i = 0; i < vertexCount; i += 3) {
        ShapeDescriptor::cpu::float3 vertex0 = vertices[i + 0];
        ShapeDescriptor::cpu::float3 vertex1 = vertices[i + 1];
        ShapeDescriptor::cpu::float3 vertex2 = vertices[i + 2];

        ShapeDescriptor::cpu::float3 normal0 = normals[i + 0];
        ShapeDescriptor::cpu::float3 normal1 = normals[i + 1];
        ShapeDescriptor::cpu::float3 normal2 = normals[i + 2];

        ShapeDescriptor::cpu::float3 normal = computeTriangleNormal(vertex0, vertex1, vertex2);

        if(normal != normal0 || normal != normal1 || normal != normal2) {
            normalsEquivalent = false;
            break;
        }
    }
    return normalsEquivalent;
}

void computeIndexBuffer(const std::vector<Vertex>& vertices,
                        std::vector<Vertex>& condensedVertices,
                        std::vector<uint32_t>& indexBuffer) {
    indexBuffer.resize(vertices.size());
    std::unordered_map<Vertex, uint32_t> seenCoordinates;
    for(uint32_t i = 0; i < vertices.size(); i++) {
        const Vertex& vertex = vertices.at(i);
        if(!seenCoordinates.contains(vertex)) {
            // Has not been seen before
            seenCoordinates.insert({vertex, condensedVertices.size()});
            condensedVertices.push_back(vertex);
        }
        indexBuffer.at(i) = seenCoordinates.at(vertex);
    }
}

void compressGeometry(std::vector<uint8_t>& compressedBuffer,
                      const std::vector<uint8_t>& geometryData,
                      uint16_t vertexSizeBytes,
                      uint32_t geometryDataEntryCount) {
    size_t vertexBufferSizeBound = meshopt_encodeVertexBufferBound(geometryDataEntryCount, vertexSizeBytes);
    compressedBuffer.resize(vertexBufferSizeBound);
    size_t compressedVertexBufferSize = meshopt_encodeVertexBuffer(compressedBuffer.data(), compressedBuffer.size(), geometryData.data(), geometryDataEntryCount, vertexSizeBytes);
    compressedBuffer.resize(compressedVertexBufferSize);
}

void compressIndexBuffer(std::vector<uint8_t>& compressedIndexBuffer,
                         std::vector<uint32_t>& indexBuffer,
                         uint32_t vertexCount) {
    size_t verticesToPad = (3 - (vertexCount % 3)) % 3;
    size_t paddedIndexCount = vertexCount + verticesToPad;
    if(verticesToPad != 0) {
        // library assumes triangles. Need to invent some additional indices for point clouds and the like
        indexBuffer.resize(paddedIndexCount);
    }

    size_t indexBufferSizeBound = meshopt_encodeIndexBufferBound(indexBuffer.size(), vertexCount);
    compressedIndexBuffer.resize(indexBufferSizeBound);
    size_t compressedIndexBufferSize = meshopt_encodeIndexBuffer(compressedIndexBuffer.data(), compressedIndexBuffer.size(), indexBuffer.data(), indexBuffer.size());
    compressedIndexBuffer.resize(compressedIndexBufferSize);
}

void dumpCompressedGeometry(const ShapeDescriptor::cpu::float3* vertices,
                            const ShapeDescriptor::cpu::float3* normals,
                            const ShapeDescriptor::cpu::uchar4* vertexColours,
                            const uint32_t vertexCount, // Note: code relies on that vertex count is 32 bit
                            const std::filesystem::path &filePath,
                            bool stripVertexColours,
                            bool isPointCloud) {
    meshopt_encodeVertexVersion(0);
    meshopt_encodeIndexVersion(1);

    bool containsNormals = normals != nullptr;
    bool containsVertexColours = vertexColours != nullptr && !stripVertexColours;

    bool originalMeshContainedNormals = containsNormals;

    // If all normals can be computed exactly based on the triangles in the mesh, we do not need to store them
    // We can just compute them when loading the mesh instead.
    bool normalsEquivalent = canMeshNormalsBeComputedExactly(vertices, normals, vertexCount);

    if(normalsEquivalent) {
        // Do not save any normals when we can compute them perfectly
        containsNormals = false;
    }

    std::vector<Vertex> geometry(vertexCount);
    for(uint32_t i = 0; i < vertexCount; i++) {
        Vertex vertex;
        vertex.position = vertices[i];
        if(containsNormals) {
            vertex.normal = normals[i];
        }
        if(containsVertexColours) {
            vertex.colour = vertexColours[i];
        }
        geometry.at(i) = vertex;
    }


    std::vector<Vertex> condensedGeometry;
    std::vector<uint32_t> indexBuffer;

    computeIndexBuffer(geometry, condensedGeometry, indexBuffer);

    uint32_t stride = sizeof(ShapeDescriptor::cpu::float3)
            + (containsNormals ? sizeof(ShapeDescriptor::cpu::float3) : 0)
            + (containsVertexColours ? sizeof(ShapeDescriptor::cpu::uchar4) : 0);

    std::vector<uint8_t> compactedGeometryData(stride * condensedGeometry.size());
    for(uint32_t i = 0; i < condensedGeometry.size(); i++) {
        uint8_t* valuePointer = compactedGeometryData.data() + stride * i;
        Vertex vertex = condensedGeometry.at(i);
        *reinterpret_cast<ShapeDescriptor::cpu::float3*>(valuePointer) = vertex.position;
        valuePointer += sizeof(ShapeDescriptor::cpu::float3);
        if(containsNormals) {
            *reinterpret_cast<ShapeDescriptor::cpu::float3*>(valuePointer) = vertex.normal;
            valuePointer += sizeof(ShapeDescriptor::cpu::float3);
        }
        if(containsVertexColours) {
            *reinterpret_cast<ShapeDescriptor::cpu::uchar4*>(valuePointer) = vertex.colour;
            valuePointer += sizeof(ShapeDescriptor::cpu::uchar4);
        }
    }

    std::vector<uint8_t> compressedGeometryBuffer;
    compressGeometry(compressedGeometryBuffer, compactedGeometryData, stride, condensedGeometry.size());

    std::vector<uint8_t> compressedIndexBuffer;
    compressIndexBuffer(compressedIndexBuffer, indexBuffer, condensedGeometry.size());




    // Header consists of:
    // - 8 byte magic
    // - 4 byte file version
    // - 4 byte flags
    // - 4 byte uncondensed/original vertex count
    // - 2 x 4 byte condensed vertex/normal buffers lengths
    // - 4 x 4 byte compressed vertex/normal/vertex_index/normal_index buffer sizes in bytes
    // - 4 byte compressed vertex colour buffer
    // (note: 32-bit colour is the same size as the index buffer, and therefore does not save on file size. It thus does not have an index buffer)
    const uint32_t headerSize = sizeof(uint64_t) + 6 * sizeof(uint32_t);
    const uint32_t geometryBufferSize = compressedGeometryBuffer.size();
    const uint32_t indexBufferSize = compressedIndexBuffer.size();
    std::vector<uint8_t> fileBuffer(headerSize + geometryBufferSize + indexBufferSize);
    uint8_t* bufferPointer = fileBuffer.data();



    // header: magic
    const uint64_t magic = 0x4F45474853454D43;
    bufferPointer = write(magic, bufferPointer);

    // header: version
    const uint32_t fileSpecVersion = 2;
    bufferPointer = write(fileSpecVersion, bufferPointer);

    // header: flags
    const uint32_t flagContainsNormals = containsNormals ? 1 : 0;
    const uint32_t flagContainsVertexColours = containsVertexColours ? 2 : 0;
    const uint32_t flagIsPointCloud = isPointCloud ? 4 : 0;
    const uint32_t flagNormalsWereRemoved = originalMeshContainedNormals ? 8 : 0;
    const uint32_t flags =
              flagContainsNormals
            | flagContainsVertexColours
            | flagIsPointCloud
            | flagNormalsWereRemoved;
    bufferPointer = write(flags, bufferPointer);

    // header: uncondensed vertex count
    bufferPointer = write(vertexCount, bufferPointer);

    // header: condensed vertex count
    const uint32_t condensedVertexCount = condensedGeometry.size();
    bufferPointer = write(condensedVertexCount, bufferPointer);

    // header: compressed buffer sizes
    bufferPointer = write(geometryBufferSize, bufferPointer);
    bufferPointer = write(indexBufferSize, bufferPointer);

    // contents: geometry data
    bufferPointer = write(compressedGeometryBuffer, bufferPointer);
    bufferPointer = write(compressedIndexBuffer, bufferPointer);

    assert(bufferPointer == fileBuffer.data() + fileBuffer.size());

    ShapeDescriptor::utilities::writeCompressedFile((char*) fileBuffer.data(), fileBuffer.size(), filePath, 4);
}

void ShapeDescriptor::utilities::writeCompressedGeometryFile(const ShapeDescriptor::cpu::Mesh &mesh, const std::filesystem::path &filePath, bool stripVertexColours) {
    dumpCompressedGeometry(mesh.vertices, mesh.normals, mesh.vertexColours, mesh.vertexCount, filePath, stripVertexColours, false);
}

void ShapeDescriptor::utilities::writeCompressedGeometryFile(const ShapeDescriptor::cpu::PointCloud &cloud, const std::filesystem::path &filePath, bool stripVertexColours) {
    dumpCompressedGeometry(cloud.vertices, cloud.normals, cloud.vertexColours, cloud.pointCount, filePath, stripVertexColours, true);
}
