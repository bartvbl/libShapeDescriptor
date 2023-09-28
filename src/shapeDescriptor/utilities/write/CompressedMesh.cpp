#include "CompressedMesh.h"
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

void ShapeDescriptor::utilities::writeCompressedMesh(const ShapeDescriptor::cpu::Mesh &mesh, const std::filesystem::path &filePath, bool stripVertexColours) {
    // limits supported number of triangles per triangle strip to 2B
    const uint32_t TRIANGLE_STRIP_END_FLAG = 0x1U << 31;

    meshopt_encodeVertexVersion(0);
    meshopt_encodeIndexVersion(1);

    bool containsNormals = mesh.normals != nullptr;
    bool containsVertexColours = mesh.vertexColours != nullptr && !stripVertexColours;

    bool originalMeshContainedNormals = containsNormals;

    // If all normals can be computed exactly based on the triangles in the mesh, we do not need to store them
    // We can just compute them when loading the mesh instead.
    bool normalsEquivalent = true;
    for(size_t i = 0; i < mesh.vertexCount; i += 3) {
        ShapeDescriptor::cpu::float3 vertex0 = mesh.vertices[i + 0];
        ShapeDescriptor::cpu::float3 vertex1 = mesh.vertices[i + 1];
        ShapeDescriptor::cpu::float3 vertex2 = mesh.vertices[i + 2];

        ShapeDescriptor::cpu::float3 normal0 = mesh.normals[i + 0];
        ShapeDescriptor::cpu::float3 normal1 = mesh.normals[i + 1];
        ShapeDescriptor::cpu::float3 normal2 = mesh.normals[i + 2];

        ShapeDescriptor::cpu::float3 normal = computeTriangleNormal(vertex0, vertex1, vertex2);

        if(normal != normal0 || normal != normal1 || normal != normal2) {
            normalsEquivalent = false;
            break;
        }
    }

    if(normalsEquivalent) {
        // Do not save any normals when we can compute them perfectly
        containsNormals = false;
    }

    std::vector<ShapeDescriptor::cpu::float3> condensedVertices;
    std::vector<uint32_t> vertexIndexBuffer(mesh.vertexCount);
    std::unordered_set<ShapeDescriptor::cpu::float3> seenUniqueVertices;
    std::unordered_map<ShapeDescriptor::cpu::float3, uint32_t> seenVerticesIndex;

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

    meshopt_optimizeVertexCacheStrip(vertexIndexBuffer.data(), vertexIndexBuffer.data(), vertexIndexBuffer.size(), condensedVertices.size());
    meshopt_optimizeVertexFetch(condensedVertices.data(), vertexIndexBuffer.data(), vertexIndexBuffer.size(), condensedVertices.data(), condensedVertices.size(), sizeof(ShapeDescriptor::cpu::float3));

    // Increases file size, even though it reduces the length of the index buffer rather drastically. LZMA2 does much better,
    //std::vector<unsigned int> vertexIndexBuffer(meshopt_stripifyBound(nonStrippedVertexIndexBuffer.size()));
    //size_t stripifiedVertexIndexBufferSize = meshopt_stripify(vertexIndexBuffer.data(), nonStrippedVertexIndexBuffer.data(), nonStrippedVertexIndexBuffer.size(), condensedVertices.size(), ~0u);
    //vertexIndexBuffer.resize(stripifiedVertexIndexBufferSize);

    size_t vertexBufferSizeBound = meshopt_encodeVertexBufferBound(condensedVertices.size(), sizeof(ShapeDescriptor::cpu::float3));
    std::vector<unsigned char> compressedVertexBuffer(vertexBufferSizeBound);
    size_t compressedVertexBufferSize = meshopt_encodeVertexBuffer(compressedVertexBuffer.data(), compressedVertexBuffer.size(), condensedVertices.data(), condensedVertices.size(), sizeof(ShapeDescriptor::cpu::float3));
    compressedVertexBuffer.resize(compressedVertexBufferSize);

    size_t indexBufferSizeBound = meshopt_encodeIndexBufferBound(vertexIndexBuffer.size(), mesh.vertexCount);
    std::vector<unsigned char> compressedIndexBuffer(indexBufferSizeBound);
    size_t compressedIndexBufferSize = meshopt_encodeIndexBuffer(compressedIndexBuffer.data(), compressedIndexBuffer.size(), vertexIndexBuffer.data(), vertexIndexBuffer.size());
    compressedIndexBuffer.resize(compressedIndexBufferSize);


    // -- Compressing normals --

    std::vector<ShapeDescriptor::cpu::float3> condensedNormals;
    std::vector<uint32_t> normalIndexBuffer;
    std::unordered_set<ShapeDescriptor::cpu::float3> seenUniqueNormals;
    std::unordered_map<ShapeDescriptor::cpu::float3, uint32_t> seenNormalsIndex;

    condensedVertices.reserve(mesh.vertexCount);
    if(containsNormals) {
        condensedNormals.reserve(mesh.vertexCount);
        normalIndexBuffer.resize(mesh.vertexCount);
    }

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

        meshopt_optimizeVertexCacheStrip(normalIndexBuffer.data(), normalIndexBuffer.data(), normalIndexBuffer.size(), condensedNormals.size());
        meshopt_optimizeVertexFetch(condensedNormals.data(), normalIndexBuffer.data(), normalIndexBuffer.size(), condensedNormals.data(), condensedNormals.size(), sizeof(ShapeDescriptor::cpu::float3));

        //std::vector<unsigned int> normalIndexBuffer(meshopt_stripifyBound(nonStrippedNormalIndexBuffer.size()));
        //size_t stripifiedNormalIndexBufferSize = meshopt_stripify(normalIndexBuffer.data(), nonStrippedNormalIndexBuffer.data(), nonStrippedNormalIndexBuffer.size(), condensedNormals.size(), ~0u);
        //normalIndexBuffer.resize(stripifiedNormalIndexBufferSize);

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
    std::vector<unsigned char> compressedColourBuffer;

    if(containsVertexColours) {
        unsigned int nextColourID = 0;
        std::map<ShapeDescriptor::cpu::uchar4, unsigned int> colourPalette;
        std::vector<ShapeDescriptor::cpu::uchar4> colours;
        std::vector<unsigned int> paletteIndices(mesh.vertexCount);
        for(size_t i = 0; i < mesh.vertexCount; i++) {
            ShapeDescriptor::cpu::uchar4 vertexColour = mesh.vertexColours[i];
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
            std::copy(mesh.vertexColours, mesh.vertexColours + mesh.vertexCount, reinterpret_cast<ShapeDescriptor::cpu::uchar4*>(compressedColourBuffer.data() + headerSize));
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
    const uint32_t headerSize = 6 * sizeof(uint64_t) + 5 * sizeof(uint32_t);
    const size_t vertexSize = compressedVertexBufferSize;
    const size_t normalSize = compressedNormalBufferSize;
    const size_t colourSize = compressedColourBuffer.size();
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
    const uint32_t flagNormalsWereRemoved = originalMeshContainedNormals ? 4 : 0;
    const uint32_t flags = flagContainsNormals | flagContainsVertexColours | flagNormalsWereRemoved;
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
        std::copy(compressedColourBuffer.begin(), compressedColourBuffer.end(), bufferPointer);
        bufferPointer += compressedColourBuffer.size();
    }

    assert(bufferPointer == fileBuffer.data() + fileBuffer.size());

    ShapeDescriptor::utilities::writeCompressedFile((char*) fileBuffer.data(), fileBuffer.size(), filePath, 4);
}
