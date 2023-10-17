#include <shapeDescriptor/shapeDescriptor.h>
#include <meshoptimizer.h>
#include <iostream>

uint32_t readUint32(char*& bufferPointer) {
    uint32_t value = *reinterpret_cast<uint32_t*>(bufferPointer);
    bufferPointer += sizeof(uint32_t);
    return value;
}

uint64_t readUint64(char*& bufferPointer) {
    uint64_t value = *reinterpret_cast<uint64_t*>(bufferPointer);
    bufferPointer += sizeof(uint64_t);
    return value;
}

void decompressGeometryBuffer(size_t extractedCount, ShapeDescriptor::cpu::float3* destination, unsigned char* compressedVertexBuffer, size_t compressedVertexBufferSize) {
    int resvb = meshopt_decodeVertexBuffer(destination, extractedCount, sizeof(ShapeDescriptor::cpu::float3), compressedVertexBuffer, compressedVertexBufferSize);
    assert(resvb == 0);
}

void decompressGeometryWithIndexBuffer(size_t extractedCount, size_t condensedCount, ShapeDescriptor::cpu::float3* destination, unsigned char* compressedVertexBuffer, size_t compressedVertexBufferSize, unsigned char* compressedIndexBuffer, size_t compressedIndexBufferSize) {
    std::vector<ShapeDescriptor::cpu::float3> condensedVertices(condensedCount);

    size_t verticesToPad = (3 - (extractedCount % 3)) % 3;
    size_t paddedIndexCount = extractedCount + verticesToPad;
    // library assumes triangles, so the index buffer was padded when writing the file
    // we need to include that padding while decompressing the index buffer
    // However, we just ignore the final indices (if any) afterwards
    std::vector<unsigned int> vertexIndexBuffer(paddedIndexCount);

    int resvb = meshopt_decodeVertexBuffer(condensedVertices.data(), condensedCount, sizeof(ShapeDescriptor::cpu::float3), compressedVertexBuffer, compressedVertexBufferSize);
    int resib = meshopt_decodeIndexBuffer(vertexIndexBuffer.data(), paddedIndexCount, compressedIndexBuffer, compressedIndexBufferSize);
    assert(resvb == 0 && resib == 0);
    

    for(uint32_t i = 0; i < extractedCount; i++) {
        uint32_t index = vertexIndexBuffer.at(i);
        destination[i] = condensedVertices.at(index);
    }
}

void readGeometryDataFromFile(const std::filesystem::path &filePath,
                              ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::float3>& vertices,
                              ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::float3>& normals,
                              ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::uchar4>& vertexColours,
                              bool expectMeshInFile) {
    meshopt_encodeVertexVersion(0);
    meshopt_encodeIndexVersion(1);

    std::vector<char> fileContents = ShapeDescriptor::readCompressedFile(filePath, 4);
    char* bufferPointer = fileContents.data();

    // Read header
    // header: magic
    uint64_t magic = readUint64(bufferPointer);
    if(magic != 0x4F45474853454D43) {
        throw std::runtime_error("Invalid magic bytes detected when reading compressed file from (the file has the wrong format or may be corrupt): " + filePath.string());
    }

    uint32_t fileVersion = readUint32(bufferPointer);

    // header: flags
    uint32_t flags = readUint32(bufferPointer);
    bool flagContainsNormals = (flags & 1) != 0;
    bool flagContainsVertexColours = (flags & 2) != 0;
    bool flagIsPointCloud = (flags & 4) != 0;
    bool flagOriginalMeshContainedNormals = (flags & 8) != 0;
    bool flagUseVertexIndexBuffer = (flags & 16) != 0;
    bool flagUseNormalIndexBuffer = (flags & 32) != 0;

    bool normalsWereRemoved = flagOriginalMeshContainedNormals && !flagContainsNormals;

    if(expectMeshInFile == flagIsPointCloud) {
        throw std::runtime_error("Error while reading file: " + filePath.string() + "\nFile was expected to contain a " + (expectMeshInFile ? "mesh" : "point cloud") + ", but in reality contains a " + (flagIsPointCloud ? "point cloud" : "mesh") + ".");
    }

    // header: uncondensed vertex count
    uint32_t vertexCount = readUint32(bufferPointer);

    // Allocate buffers
    vertices = ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::float3>(vertexCount);
    if(flagContainsNormals || normalsWereRemoved) {
        normals = ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::float3>(vertexCount);
    } else {
        normals = {0, nullptr};
    }
    if(flagContainsVertexColours) {
        vertexColours = ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::uchar4>(vertexCount);
    } else {
        vertexColours = {0, nullptr};
    }

    // header: condensed buffer lengths
    uint32_t condensedVertexCount = readUint32(bufferPointer);
    uint32_t condensedNormalCount = readUint32(bufferPointer);

    // header: compressed vertex/normal/vertex_index/vertex_normal buffer sizes
    uint32_t compressedVertexBufferSize = readUint32(bufferPointer);
    uint32_t compressedNormalBufferSize = readUint32(bufferPointer);
    uint32_t compressedVertexIndexBufferSize = readUint32(bufferPointer);
    uint32_t compressedNormalIndexBufferSize = readUint32(bufferPointer);
    uint32_t displacementBufferSize = readUint32(bufferPointer);

    // We can read the compressed buffers directly from the file's buffer
    char* compressedVertexBuffer = bufferPointer;
    bufferPointer += compressedVertexBufferSize;
    char* compressedNormalBuffer = bufferPointer;
    bufferPointer += compressedNormalBufferSize;
    char* compressedVertexIndexBuffer = bufferPointer;
    bufferPointer += compressedVertexIndexBufferSize;
    char* compressedNormalIndexBuffer = bufferPointer;
    bufferPointer += compressedNormalIndexBufferSize;
    char* displacementBuffer = bufferPointer;
    bufferPointer += displacementBufferSize;

    // Read vertices
    if(flagUseVertexIndexBuffer) {
        decompressGeometryWithIndexBuffer(vertexCount,
                                          condensedVertexCount, vertices.content,
                                          reinterpret_cast<uint8_t*>(compressedVertexBuffer), compressedVertexBufferSize,
                                          reinterpret_cast<uint8_t*>(compressedVertexIndexBuffer), compressedVertexIndexBufferSize);
    } else {
        decompressGeometryBuffer(vertexCount, vertices.content, reinterpret_cast<uint8_t*>(compressedVertexBuffer), compressedVertexBufferSize);
    }

    if(flagUseVertexIndexBuffer && normalsWereRemoved && displacementBufferSize > 0) {
        for(uint32_t i = 0; i < vertexCount; i+=3) {
            uint32_t triangleIndex = i / 3;
            uint8_t rotation = (displacementBuffer[triangleIndex / 4] >> (6 - 2 * (triangleIndex % 4)) & 0b11);
            if(rotation == 2) {
                ShapeDescriptor::cpu::float3 tempNormal = vertices.content[i];
                vertices.content[i] = vertices.content[i + 1];
                vertices.content[i + 1] = vertices.content[i + 2];
                vertices.content[i + 2] = tempNormal;
            }
            if(rotation == 1) {
                ShapeDescriptor::cpu::float3 tempNormal = vertices.content[i + 2];
                vertices.content[i + 2] = vertices.content[i + 1];
                vertices.content[i + 1] = vertices.content[i];
                vertices.content[i] = tempNormal;
            }
        }
    }

    if(flagContainsNormals) {
        if(flagUseNormalIndexBuffer) {
            decompressGeometryWithIndexBuffer(vertexCount,
                                              condensedNormalCount, normals.content,
                                              reinterpret_cast<uint8_t*>(compressedNormalBuffer), compressedNormalBufferSize,
                                              reinterpret_cast<uint8_t*>(compressedNormalIndexBuffer), compressedNormalIndexBufferSize);
        } else {
            decompressGeometryBuffer(vertexCount, normals.content, reinterpret_cast<uint8_t*>(compressedNormalBuffer), compressedNormalBufferSize);
        }

    // Normals were present, but removed because they could be calculated exactly.
    // We are therefore expected to recompute them here
    } else if(normalsWereRemoved) {
        for(uint32_t i = 0; i < vertexCount; i += 3) {
            ShapeDescriptor::cpu::float3 vertex0 = vertices.content[i];
            ShapeDescriptor::cpu::float3 vertex1 = vertices.content[i + 1];
            ShapeDescriptor::cpu::float3 vertex2 = vertices.content[i + 2];
            ShapeDescriptor::cpu::float3 normal = ShapeDescriptor::computeTriangleNormal(vertex0, vertex1, vertex2);
            normals.content[i] = normal;
            normals.content[i + 1] = normal;
            normals.content[i + 2] = normal;
        }
    }

    if(flagUseVertexIndexBuffer && flagUseNormalIndexBuffer && !normalsWereRemoved && displacementBufferSize > 0) {
        for(uint32_t i = 0; i < vertexCount; i+=3) {
            uint32_t triangleIndex = i / 3;
            uint8_t rotation = (displacementBuffer[triangleIndex / 4] >> (6 - 2 * (triangleIndex % 4)) & 0b11);
            if(rotation == 2) {
                ShapeDescriptor::cpu::float3 tempNormal = normals.content[i];
                normals.content[i] = normals.content[i + 1];
                normals.content[i + 1] = normals.content[i + 2];
                normals.content[i + 2] = tempNormal;
            }
            if(rotation == 1) {
                ShapeDescriptor::cpu::float3 tempNormal = normals.content[i + 2];
                normals.content[i + 2] = normals.content[i + 1];
                normals.content[i + 1] = normals.content[i];
                normals.content[i] = tempNormal;
            }
        }
    }

    if(flagContainsVertexColours) {

    }

}

ShapeDescriptor::cpu::Mesh
ShapeDescriptor::loadMeshFromCompressedGeometryFile(const std::filesystem::path &filePath) {
    ShapeDescriptor::cpu::Mesh mesh;
    ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::float3> vertices;
    ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::float3> normals;
    ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::uchar4> vertexColours;
    readGeometryDataFromFile(filePath, vertices, normals, vertexColours, true);

    mesh.vertexCount = vertices.length;
    mesh.vertices = vertices.content;
    mesh.normals = normals.content;
    mesh.vertexColours = vertexColours.content;

    return mesh;
}

ShapeDescriptor::cpu::PointCloud
ShapeDescriptor::readPointCloudFromCompressedGeometryFile(const std::filesystem::path &filePath) {
    ShapeDescriptor::cpu::PointCloud cloud;
    ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::float3> vertices;
    ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::float3> normals;
    ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::uchar4> vertexColours;
    readGeometryDataFromFile(filePath, vertices, normals, vertexColours, false);

    cloud.pointCount = vertices.length;
    cloud.vertices = vertices.content;
    cloud.normals = normals.content;
    cloud.vertexColours = vertexColours.content;

    return cloud;
}
