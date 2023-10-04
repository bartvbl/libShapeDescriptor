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

void computeIndexBuffer(const ShapeDescriptor::cpu::float3* coordinates,
                        uint32_t vertexCount,
                        std::vector<ShapeDescriptor::cpu::float3>& condensedGeometryBuffer,
                        std::vector<uint32_t>& indexBuffer) {
    indexBuffer.resize(vertexCount);
    std::unordered_map<ShapeDescriptor::cpu::float3, uint32_t> seenCoordinates;
    for(uint32_t i = 0; i < vertexCount; i++) {
        const ShapeDescriptor::cpu::float3 coordinate = coordinates[i];
        if(!seenCoordinates.contains(coordinate)) {
            // Has not been seen before
            seenCoordinates.insert({coordinate, condensedGeometryBuffer.size()});
            condensedGeometryBuffer.push_back(coordinate);
        }
        indexBuffer.at(i) = seenCoordinates.at(coordinate);
    }
}

void compressGeometry(std::vector<unsigned char>& compressedBuffer,
                      const ShapeDescriptor::cpu::float3* geometryData,
                      uint32_t geometryDataEntryCount) {
    size_t vertexBufferSizeBound = meshopt_encodeVertexBufferBound(geometryDataEntryCount, sizeof(ShapeDescriptor::cpu::float3));
    compressedBuffer.resize(vertexBufferSizeBound);
    size_t compressedVertexBufferSize = meshopt_encodeVertexBuffer(compressedBuffer.data(), compressedBuffer.size(), geometryData, geometryDataEntryCount, sizeof(ShapeDescriptor::cpu::float3));
    compressedBuffer.resize(compressedVertexBufferSize);
}

void compressIndexBuffer(std::vector<unsigned char>& compressedIndexBuffer,
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
                            const uint32_t vertexCount, // Note: important vertex count is 32 bit
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
        std::cout << std::endl << filePath.string() + " -> equivalent normals" << std::endl;
        containsNormals = false;
    }

    std::vector<ShapeDescriptor::cpu::float3> condensedVertices;
    std::vector<uint32_t> vertexIndexBuffer;

    // -- Compressing vertex positions --

    if(!isPointCloud) {
        computeIndexBuffer(vertices, vertexCount, condensedVertices, vertexIndexBuffer);
    }

    // Creating triangle strips yields worse file sizes due to worse LZMA2 compression

    // Some meshes such as point clouds only really have unique vertices. It thus does not always make sense to store an index buffer
    // This is a heuristic, but not a correct computation of what the final size will look like given that we have not yet compressed it all.
    size_t vertexBufferSizeWithIndexBuffer = sizeof(ShapeDescriptor::cpu::float3) * condensedVertices.size() + sizeof(uint32_t) * vertexIndexBuffer.size();
    size_t vertexBufferSizeWithoutIndexBuffer = sizeof(ShapeDescriptor::cpu::float3) * vertexCount;
    // If this should be re-enabled, make sure to read from the raw vertex buffer when writing the file
    bool includeVertexIndexBuffer = vertexBufferSizeWithIndexBuffer < vertexBufferSizeWithoutIndexBuffer && !isPointCloud;

    std::vector<unsigned char> compressedVertexBuffer;
    compressGeometry(compressedVertexBuffer,
                     includeVertexIndexBuffer ? condensedVertices.data() : vertices,
                     includeVertexIndexBuffer ? condensedVertices.size() : vertexCount);

    std::vector<unsigned char> compressedIndexBuffer;
    if(includeVertexIndexBuffer) {
        compressIndexBuffer(compressedIndexBuffer, vertexIndexBuffer, vertexCount);
    }

    // -- Compressing normals --

    std::vector<ShapeDescriptor::cpu::float3> condensedNormals;
    std::vector<uint32_t> normalIndexBuffer;

    std::vector<unsigned char> compressedNormalBuffer;
    std::vector<unsigned char> compressedNormalIndexBuffer;

    bool includeNormalIndexBuffer = false;
    if(containsNormals) {
        computeIndexBuffer(normals, vertexCount, condensedNormals, normalIndexBuffer);

        size_t normalBufferSizeWithIndexBuffer = sizeof(ShapeDescriptor::cpu::float3) * condensedNormals.size() + sizeof(unsigned int) * normalIndexBuffer.size();
        size_t normalBufferSizeWithoutIndexBuffer = sizeof(ShapeDescriptor::cpu::float3) * vertexCount;
        includeNormalIndexBuffer = /*normalBufferSizeWithIndexBuffer < normalBufferSizeWithoutIndexBuffer &&*/ !isPointCloud;

        compressGeometry(compressedNormalBuffer,
                         includeNormalIndexBuffer ? condensedNormals.data() : normals,
                         includeNormalIndexBuffer ? condensedNormals.size() : vertexCount);

        if(includeNormalIndexBuffer) {
            compressIndexBuffer(compressedNormalBuffer, normalIndexBuffer, vertexCount);
        }
    }

    std::vector<unsigned char> compressedColourBuffer;

    if(containsVertexColours) {
        unsigned int nextColourID = 0;
        std::map<ShapeDescriptor::cpu::uchar4, unsigned int> colourPalette;
        std::vector<ShapeDescriptor::cpu::uchar4> colours;
        std::vector<unsigned int> paletteIndices(vertexCount);
        for(size_t i = 0; i < vertexCount; i++) {
            ShapeDescriptor::cpu::uchar4 vertexColour = vertexColours[i];
            if(!colourPalette.contains(vertexColour)) {
                colourPalette.insert({vertexColour, nextColourID});
                colours.push_back(vertexColour);
                nextColourID++;
            }
            paletteIndices.push_back(colourPalette.at(vertexColour));
        }

        // Buffer format:
        // 4 bytes - number of bits per colour index
        // 4 bytes - number of unique colours in palette
        // If number of bits per colour index is 32: list of 32 bit colours, one per vertex
        // Otherwise:
        // - Unique colours (number specified in header)
        // - Index buffer (x bytes per entry, as specified in header)
        const size_t headerSize = 2 * sizeof(uint32_t);
        size_t colourPaletteSize = colours.size() * sizeof(ShapeDescriptor::cpu::uchar4);

        uint32_t bitsPerColour = 32;
        if(colours.size() > 65535) {
            // When we need 32 bit per colour index, there is no point in using an index buffer
            colourPaletteSize = 0;
        } else if(colours.size() > 255) {
            bitsPerColour = 16;
        } else {
            bitsPerColour = 8;
        }
        compressedColourBuffer.resize(headerSize + colourPaletteSize + paletteIndices.size() * (bitsPerColour / 8));

        uint32_t colourCount = colours.size();
        write(colourCount, compressedColourBuffer.data() + sizeof(uint32_t));
        write(bitsPerColour, compressedColourBuffer.data());

        if(bitsPerColour == 32) {
            std::copy(vertexColours, vertexColours + vertexCount, reinterpret_cast<ShapeDescriptor::cpu::uchar4*>(compressedColourBuffer.data() + headerSize));
        } else {
            std::copy(colours.begin(), colours.end(), reinterpret_cast<ShapeDescriptor::cpu::uchar4*>(compressedColourBuffer.data() + headerSize));
        }

        if(bitsPerColour == 16) {
            uint16_t* colourBasePointer = reinterpret_cast<uint16_t*>(compressedColourBuffer.data() + headerSize + colourPaletteSize);
            for(int i = 0; i < paletteIndices.size(); i++) {
                colourBasePointer[i] = (uint16_t) paletteIndices.at(i);
            }
        } else if(bitsPerColour == 8) {
            uint8_t* colourBasePointer = reinterpret_cast<uint8_t*>(compressedColourBuffer.data() + headerSize + colourPaletteSize);
            for(int i = 0; i < paletteIndices.size(); i++) {
                colourBasePointer[i] = (uint8_t) paletteIndices.at(i);
            }
        }
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
    const uint32_t headerSize = sizeof(uint64_t) + 10 * sizeof(uint32_t);
    const uint32_t vertexSize = compressedVertexBuffer.size();
    const uint32_t normalSize = compressedNormalBuffer.size();
    const uint32_t colourSize = compressedColourBuffer.size();
    const uint32_t vertexIndexSize = compressedIndexBuffer.size();
    const uint32_t normalIndexSize = compressedNormalIndexBuffer.size();
    std::vector<uint8_t> fileBuffer(headerSize + vertexSize + normalSize + colourSize + vertexIndexSize + normalIndexSize);
    uint8_t* bufferPointer = fileBuffer.data();



    // header: magic
    const uint64_t magic = 0x4F45474853454D43;
    bufferPointer = write(magic, bufferPointer);

    // header: version
    const uint32_t fileSpecVersion = 1;
    bufferPointer = write(fileSpecVersion, bufferPointer);

    // header: flags
    const uint32_t flagContainsNormals = containsNormals ? 1 : 0;
    const uint32_t flagContainsVertexColours = containsVertexColours ? 2 : 0;
    const uint32_t flagIsPointCloud = isPointCloud ? 4 : 0;
    const uint32_t flagNormalsWereRemoved = originalMeshContainedNormals ? 8 : 0;
    const uint32_t flagVertexIndexBufferEnabled = includeVertexIndexBuffer ? 16 : 0;
    const uint32_t flagNormalIndexBufferEnabled = includeNormalIndexBuffer ? 32 : 0;
    const uint32_t flags =
              flagContainsNormals
            | flagContainsVertexColours
            | flagIsPointCloud
            | flagNormalsWereRemoved
            | flagVertexIndexBufferEnabled
            | flagNormalIndexBufferEnabled;
    bufferPointer = write(flags, bufferPointer);

    // header: uncondensed vertex count
    bufferPointer = write(vertexCount, bufferPointer);

    // header: condensed buffer lengths
    const uint32_t condensedVertexCount = condensedVertices.size();
    const uint32_t condensedNormalCount = condensedNormals.size();
    bufferPointer = write(condensedVertexCount, bufferPointer);
    bufferPointer = write(condensedNormalCount, bufferPointer);

    // header: compressed vertex/normal/vertex_index/vertex_normal buffer sizes
    bufferPointer = write(vertexSize, bufferPointer);
    bufferPointer = write(vertexIndexSize, bufferPointer);
    bufferPointer = write(normalSize, bufferPointer);
    bufferPointer = write(normalIndexSize, bufferPointer);
    bufferPointer = write(colourSize, bufferPointer);

    // contents: vertex data
    bufferPointer = write(compressedVertexBuffer, bufferPointer);
    bufferPointer = write(compressedIndexBuffer, bufferPointer);

    // contents: normal data
    if(containsNormals) {
        bufferPointer = write(compressedNormalBuffer, bufferPointer);
        bufferPointer = write(compressedNormalIndexBuffer, bufferPointer);
    }

    // contents: colour data
    if(containsVertexColours) {
        bufferPointer = write(compressedColourBuffer, bufferPointer);
    }

    assert(bufferPointer == fileBuffer.data() + fileBuffer.size());

    ShapeDescriptor::utilities::writeCompressedFile((char*) fileBuffer.data(), fileBuffer.size(), filePath, 4);
}

void ShapeDescriptor::utilities::writeCompressedGeometryFile(const ShapeDescriptor::cpu::Mesh &mesh, const std::filesystem::path &filePath, bool stripVertexColours) {
    dumpCompressedGeometry(mesh.vertices, mesh.normals, mesh.vertexColours, mesh.vertexCount, filePath, stripVertexColours, false);
}

void ShapeDescriptor::utilities::writeCompressedGeometryFile(const ShapeDescriptor::cpu::PointCloud &cloud, const std::filesystem::path &filePath, bool stripVertexColours) {
    dumpCompressedGeometry(cloud.vertices, cloud.normals, cloud.vertexColours, cloud.pointCount, filePath, stripVertexColours, true);
}
