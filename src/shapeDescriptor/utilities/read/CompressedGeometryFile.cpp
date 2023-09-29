#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/utilities/fileutils.h>
#include "CompressedGeometryFile.h"

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

void readGeometryDataFromFile(const std::filesystem::path &filePath) {
    std::vector<char> fileContents = ShapeDescriptor::utilities::readCompressedFile(filePath, 4);
    char* bufferPointer = fileContents.data();

    // Read header
    // header: magic
    uint64_t magic = readUint64(bufferPointer);
    if(magic != 0x4C53532D4D455348) {
        throw std::runtime_error("Invalid magic bytes detected when reading compressed file from (the file has the wrong format or may be corrupt): " + filePath.string());
    }

    uint32_t fileVersion = readUint32(bufferPointer);

    // header: flags
    uint32_t flags = readUint32(bufferPointer);
    bool flagContainsNormals = (flags & 1) != 0;
    bool flagContainsVertexColours = (flags & 2) != 0;
    bool flagIsPointCloud = (flags & 4) != 0;
    bool flagNormalsWereRemoved = (flags & 8) != 0;
    bool flagVertexIndexBufferEnabled = (flags & 16) != 0;
    bool flagNormalIndexBufferEnabled = (flags & 32) != 0;

    // header: uncondensed vertex count
    uint32_t vertexCount = readUint32(bufferPointer);

    // header: condensed buffer lengths
    uint32_t condensedVertexCount = readUint32(bufferPointer);
    uint32_t condensedNormalCount = readUint32(bufferPointer);

    // header: compressed vertex/normal/vertex_index/vertex_normal buffer sizes
    size_t compressedVertexBufferSize = readUint64(bufferPointer);
    size_t compressedIndexBufferSize = readUint64(bufferPointer);
    size_t compressedNormalBufferSize = readUint64(bufferPointer);
    size_t compressedNormalIndexBufferSize = readUint64(bufferPointer);
    size_t compressedColourBufferSize = readUint64(bufferPointer);

    // We can read the compressed buffers directly from the file's buffer
    char* compressedVertexIndexBuffer = bufferPointer;
    bufferPointer += compressedIndexBufferSize;
    char* compressedVertexBuffer = bufferPointer;
    bufferPointer += compressedVertexBufferSize;
    char* compressedNormalIndexBuffer = bufferPointer;
    bufferPointer += compressedNormalIndexBufferSize;
    char* compressedNormalBuffer = bufferPointer;
    bufferPointer += compressedNormalBufferSize;
    char* compressedColourBuffer = bufferPointer;
    bufferPointer += compressedColourBufferSize;

}

ShapeDescriptor::cpu::Mesh
ShapeDescriptor::utilities::readMeshFromCompressedGeometryFile(const std::filesystem::path &filePath) {





    return ShapeDescriptor::cpu::Mesh();
}

ShapeDescriptor::cpu::PointCloud
ShapeDescriptor::utilities::readPointCloudFromCompressedGeometryFile(const std::filesystem::path &filePath) {
    return ShapeDescriptor::cpu::PointCloud();
}
