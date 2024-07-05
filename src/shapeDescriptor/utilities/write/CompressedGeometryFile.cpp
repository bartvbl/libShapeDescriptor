#include <cstdint>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <meshoptimizer.h>
#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>
#include <map>

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

        ShapeDescriptor::cpu::float3 normal = ShapeDescriptor::computeTriangleNormal(vertex0, vertex1, vertex2);

        if(normal != normal0 || normal != normal1 || normal != normal2) {
            normalsEquivalent = false;
            break;
        }
    }
    return normalsEquivalent;
}

template<typename T>
void computeIndexBuffer(const T* vertices,
                        uint32_t vertexCount,
                        std::vector<T>& condensedVertices,
                        std::vector<uint32_t>& indexBuffer) {
    indexBuffer.resize(vertexCount);
    std::unordered_map<T, uint32_t> seenCoordinates;
    for(uint32_t i = 0; i < vertexCount; i++) {
        const T& vertex = vertices[i];
        if(!seenCoordinates.contains(vertex)) {
            // Has not been seen before
            seenCoordinates.insert({vertex, condensedVertices.size()});
            condensedVertices.push_back(vertex);
        }
        indexBuffer.at(i) = seenCoordinates.at(vertex);
    }
}

template<typename T>
void compressGeometry(std::vector<uint8_t>& compressedBuffer,
                      const T* geometryData,
                      uint16_t vertexSizeBytes,
                      uint32_t geometryDataEntryCount) {
    size_t vertexBufferSizeBound = meshopt_encodeVertexBufferBound(geometryDataEntryCount, vertexSizeBytes);
    compressedBuffer.resize(vertexBufferSizeBound);
    size_t compressedVertexBufferSize = meshopt_encodeVertexBuffer(compressedBuffer.data(), compressedBuffer.size(), geometryData, geometryDataEntryCount, vertexSizeBytes);
    compressedBuffer.resize(compressedVertexBufferSize);
}

void compressIndexBuffer(std::vector<uint8_t>& compressedIndexBuffer,
                         std::vector<uint32_t>& indexBuffer,
                         uint32_t vertexCount) {
    uint32_t indexCount = indexBuffer.size();
    size_t verticesToPad = (3 - (indexBuffer.size() % 3)) % 3;
    size_t paddedIndexCount = indexBuffer.size() + verticesToPad;
    if(verticesToPad != 0) {
        // library assumes triangles. Need to invent some additional indices for point clouds and the like
        indexBuffer.resize(paddedIndexCount);
        for(int i = 0; i < verticesToPad; i++) {
            indexBuffer.at(indexCount + i) = 0;
        }
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

    bool useVertexIndexBuffer = !isPointCloud;
    bool useNormalIndexBuffer = !isPointCloud;

    std::vector<uint8_t> compressedVertices;
    std::vector<uint8_t> compressedNormals;

    std::vector<uint8_t> compressedVertexIndexBuffer;
    std::vector<uint8_t> compressedNormalIndexBuffer;

    std::vector<ShapeDescriptor::cpu::float3> condensedVertices;
    std::vector<ShapeDescriptor::cpu::float3> condensedNormals;

    std::vector<uint32_t> vertexIndexBuffer;
    std::vector<uint32_t> normalIndexBuffer;



    if(useVertexIndexBuffer) {
        computeIndexBuffer(vertices, vertexCount, condensedVertices, vertexIndexBuffer);
        compressIndexBuffer(compressedVertexIndexBuffer, vertexIndexBuffer, condensedVertices.size());
        compressGeometry(compressedVertices, condensedVertices.data(), sizeof(ShapeDescriptor::cpu::float3), condensedVertices.size());
    } else {
        compressGeometry(compressedVertices, vertices, sizeof(ShapeDescriptor::cpu::float3), vertexCount);
    }

    std::vector<uint8_t> displacementBuffer;

    if(containsNormals) {
        if(useNormalIndexBuffer) {
            computeIndexBuffer(normals, vertexCount, condensedNormals, normalIndexBuffer);
            compressIndexBuffer(compressedNormalIndexBuffer, normalIndexBuffer, condensedNormals.size());
            compressGeometry(compressedNormals, condensedNormals.data(), sizeof(ShapeDescriptor::cpu::float3),condensedNormals.size());
        } else {
            compressGeometry(compressedNormals, normals, sizeof(ShapeDescriptor::cpu::float3), vertexCount);
        }
    }

    bool usesTwoIndexBuffers = useVertexIndexBuffer && useNormalIndexBuffer;
    bool removedNormals = useVertexIndexBuffer && originalMeshContainedNormals && !containsNormals;
    if(usesTwoIndexBuffers || removedNormals) {
        std::vector<uint32_t> decodedVertexIndexBuffer;
        std::vector<uint32_t> decodedNormalIndexBuffer;

        size_t verticesToPad = (3 - (vertexCount % 3)) % 3;
        uint32_t triangleCount = (vertexCount + verticesToPad) / 3;
        displacementBuffer.resize(triangleCount / 4 + (triangleCount % 4 > 0 ? 1 : 0));

        decodedVertexIndexBuffer.resize(vertexCount + verticesToPad);
        meshopt_decodeIndexBuffer(decodedVertexIndexBuffer.data(), vertexCount + verticesToPad, compressedVertexIndexBuffer.data(), compressedVertexIndexBuffer.size());

        if(!removedNormals) {
            decodedNormalIndexBuffer.resize(vertexCount + verticesToPad);
            meshopt_decodeIndexBuffer(decodedNormalIndexBuffer.data(), vertexCount + verticesToPad, compressedNormalIndexBuffer.data(), compressedNormalIndexBuffer.size());
        }

        for(uint32_t i = 0; i < vertexCount; i+=3) {
            uint32_t baseVertex0 = vertexIndexBuffer.at(i);
            uint32_t baseVertex1 = vertexIndexBuffer.at(i + 1);
            uint32_t baseVertex2 = vertexIndexBuffer.at(i + 2);

            uint32_t movedVertex0 = decodedVertexIndexBuffer.at(i);
            uint32_t movedVertex1 = decodedVertexIndexBuffer.at(i + 1);
            uint32_t movedVertex2 = decodedVertexIndexBuffer.at(i + 2);

            uint8_t vertexRotation = 0;
            if(baseVertex0 == movedVertex0 && baseVertex1 == movedVertex1 && baseVertex2 == movedVertex2) {
                vertexRotation = 0;
            } else if(baseVertex0 == movedVertex2 && baseVertex1 == movedVertex0 && baseVertex2 == movedVertex1) {
                vertexRotation = 2;
            } else if(baseVertex0 == movedVertex1 && baseVertex1 == movedVertex2 && baseVertex2 == movedVertex0) {
                vertexRotation = 1;
            } else {
                throw std::runtime_error("UNKNOWN VERTEX ORDER!");
            }

            uint8_t normalRotation = 0;
            if(!removedNormals) {
                uint32_t baseNormal0 = normalIndexBuffer.at(i);
                uint32_t baseNormal1 = normalIndexBuffer.at(i + 1);
                uint32_t baseNormal2 = normalIndexBuffer.at(i + 2);

                uint32_t movedNormal0 = decodedNormalIndexBuffer.at(i);
                uint32_t movedNormal1 = decodedNormalIndexBuffer.at(i + 1);
                uint32_t movedNormal2 = decodedNormalIndexBuffer.at(i + 2);

                if(baseNormal0 == movedNormal0 && baseNormal1 == movedNormal1 && baseNormal2 == movedNormal2) {
                    normalRotation = 0;
                } else if(baseNormal0 == movedNormal2 && baseNormal1 == movedNormal0 && baseNormal2 == movedNormal1) {
                    normalRotation = 2;
                } else if(baseNormal0 == movedNormal1 && baseNormal1 == movedNormal2 && baseNormal2 == movedNormal0) {
                    normalRotation = 1;
                } else {
                    throw std::runtime_error("UNKNOWN NORMAL ORDER!");
                }
            }

            int difference = int(vertexRotation) - int(normalRotation);
            if(removedNormals) {
                difference = (3 - difference) % 3;
            }
            uint8_t rotation = uint8_t(difference + (difference < 0 ? 3 : 0));
            assert(rotation >= 0 && rotation < 3);

            uint32_t triangleIndex = i / 3;
            displacementBuffer.at(triangleIndex / 4) |= (rotation << (6 - 2 * (triangleIndex % 4)));
        }
    }




    // Header consists of:
    // - 8 byte magic
    // - 4 byte file version
    // - 4 byte flags
    // - 4 byte uncondensed/original vertex count
    // - 2 x 4 byte condensed vertex/normal buffers lengths
    // - 4 x 4 byte compressed vertex/normal/vertex_index/normal_index buffer sizes in bytes
    // - 4 byte compressed vertex colour buffer
    // (note: 32-bit colour is the same size as the index buffer, and therefore does not save on file size. It thus does not have an index buffer)
    const uint32_t headerSize = sizeof(uint64_t) + 10 * sizeof(uint32_t);
    const uint32_t vertexBufferSize = compressedVertices.size();
    const uint32_t normalBufferSize = compressedNormals.size();
    const uint32_t vertexIndexBufferSize = compressedVertexIndexBuffer.size();
    const uint32_t normalIndexBufferSize = compressedNormalIndexBuffer.size();
    const uint32_t displacementBufferSize = displacementBuffer.size();
    std::vector<uint8_t> fileBuffer(headerSize + vertexBufferSize + normalBufferSize + vertexIndexBufferSize + normalIndexBufferSize + displacementBufferSize);
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
    const uint32_t flagOriginalMeshContainedNormals = originalMeshContainedNormals ? 8 : 0;
    const uint32_t flagUseVertexIndexBuffer = useVertexIndexBuffer ? 16 : 0;
    const uint32_t flagUseNormalIndexBuffer = useNormalIndexBuffer ? 32 : 0;
    const uint32_t flags =
              flagContainsNormals
            | flagContainsVertexColours
            | flagIsPointCloud
            | flagOriginalMeshContainedNormals
            | flagUseVertexIndexBuffer
            | flagUseNormalIndexBuffer;
    bufferPointer = write(flags, bufferPointer);

    // header: uncondensed vertex count
    bufferPointer = write(vertexCount, bufferPointer);

    // header: condensed vertex count
    const uint32_t condensedVertexCount = condensedVertices.size();
    bufferPointer = write(condensedVertexCount, bufferPointer);
    const uint32_t condensedNormalCount = condensedNormals.size();
    bufferPointer = write(condensedNormalCount, bufferPointer);

    // header: compressed buffer sizes
    bufferPointer = write(vertexBufferSize, bufferPointer);
    bufferPointer = write(normalBufferSize, bufferPointer);
    bufferPointer = write(vertexIndexBufferSize, bufferPointer);
    bufferPointer = write(normalIndexBufferSize, bufferPointer);
    bufferPointer = write(displacementBufferSize, bufferPointer);

    // contents: geometry data
    bufferPointer = write(compressedVertices, bufferPointer);
    bufferPointer = write(compressedNormals, bufferPointer);
    bufferPointer = write(compressedVertexIndexBuffer, bufferPointer);
    bufferPointer = write(compressedNormalIndexBuffer, bufferPointer);
    bufferPointer = write(displacementBuffer, bufferPointer);

    assert(bufferPointer == fileBuffer.data() + fileBuffer.size());

    // Done with 1 thread on purpose!
    // Using multithreaded makes written files nondeterministic.
    ShapeDescriptor::writeCompressedFile((char*) fileBuffer.data(), fileBuffer.size(), filePath, 1);
}

void ShapeDescriptor::writeCompressedGeometryFile(const ShapeDescriptor::cpu::Mesh &mesh, const std::filesystem::path &filePath, bool stripVertexColours) {
    dumpCompressedGeometry(mesh.vertices, mesh.normals, mesh.vertexColours, mesh.vertexCount, filePath, stripVertexColours, false);
}

void ShapeDescriptor::writeCompressedGeometryFile(const ShapeDescriptor::cpu::PointCloud &cloud, const std::filesystem::path &filePath, bool stripVertexColours) {
    dumpCompressedGeometry(cloud.vertices, cloud.normals, cloud.vertexColours, cloud.pointCount, filePath, stripVertexColours, true);
}
