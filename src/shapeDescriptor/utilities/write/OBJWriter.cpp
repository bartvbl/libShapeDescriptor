#include "shapeDescriptor/shapeDescriptor.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>
#include <sstream>
#include <unordered_set>
#include <unordered_map>

void dumpMesh(ShapeDescriptor::cpu::Mesh mesh, const std::filesystem::path &outputFilePath, size_t highlightStartVertex, size_t highlightEndVertex,
        bool useCustomTextureMap, ShapeDescriptor::cpu::array<float2> vertexTextureCoordinates, std::string textureMapPath) {

    bool hasHighlightsEnabled =
            (highlightStartVertex > 0 && highlightStartVertex <= mesh.vertexCount) ||
            (highlightEndVertex > 0 && highlightEndVertex <= mesh.vertexCount);
    bool hasNormalsEnabled = mesh.normals != nullptr;

    std::ofstream outputFile;
    outputFile.open(outputFilePath);
    std::filesystem::path mtlLibPath = outputFilePath;
    const std::string materialLibFileName = mtlLibPath.filename().replace_extension(".mtl");

    std::stringstream fileContents;

    if(hasHighlightsEnabled || useCustomTextureMap) {
        std::filesystem::path materialLibPath =
                outputFilePath.parent_path() / materialLibFileName;

        std::ofstream materialFile;
        materialFile.open(materialLibPath.string());

        materialFile << "newmtl regularMaterial" << std::endl;
        materialFile << "Ns 10.0000" << std::endl;
        materialFile << "Ni 1.5000" << std::endl;
        materialFile << "d 1.0000" << std::endl;
        materialFile << "Tr 0.0000" << std::endl;
        materialFile << "Tf 1.0000 1.0000 1.0000" << std::endl;
        materialFile << "illum 2" << std::endl;
        materialFile << "Ka 0.0000 0.0000 0.0000" << std::endl;
        materialFile << "Kd 0.6000 0.6000 0.6000" << std::endl;
        materialFile << "Ks 0.6000 0.6000 0.6000" << std::endl;
        materialFile << "Ke 0.0000 0.0000 0.0000" << std::endl;

        if(useCustomTextureMap) {
            materialFile << "map_Kd " << textureMapPath << std::endl;
        } else {
            materialFile << std::endl;
            materialFile << "newmtl highlightMaterial" << std::endl;
            materialFile << "Ns 10.0000" << std::endl;
            materialFile << "Ni 1.5000" << std::endl;
            materialFile << "d 1.0000" << std::endl;
            materialFile << "Tr 0.0000" << std::endl;
            materialFile << "Tf 1.0000 1.0000 1.0000" << std::endl;
            materialFile << "illum 2" << std::endl;
            materialFile << "Ka 0.0000 0.0000 0.0000" << std::endl;
            materialFile << "Kd 0.6000 0.0000 0.0000" << std::endl;
            materialFile << "Ks 0.6000 0.6000 0.6000" << std::endl;
            materialFile << "Ke 0.0000 0.0000 0.0000" << std::endl;
        }

        materialFile.close();

        fileContents << "mtllib " << materialLibFileName << std::endl;

        fileContents << std::endl;
    }

    std::vector<ShapeDescriptor::cpu::float3> condensedVertices;
    std::vector<unsigned int> vertexIndexBuffer(mesh.vertexCount);
    std::vector<ShapeDescriptor::cpu::float3> condensedNormals;
    std::vector<unsigned int> normalIndexBuffer(mesh.vertexCount);

    condensedVertices.reserve(mesh.vertexCount);
    condensedNormals.reserve(mesh.vertexCount);

    std::unordered_map<ShapeDescriptor::cpu::float3, unsigned int> seenVerticesIndex;

    std::unordered_map<ShapeDescriptor::cpu::float3, unsigned int> seenNormalsIndex;

    for(unsigned int i = 0; i < mesh.vertexCount; i++) {
        const ShapeDescriptor::cpu::float3 vertex = mesh.vertices[i];
        if(!seenVerticesIndex.contains(vertex)) {
            // Vertex has not been seen before
            seenVerticesIndex.insert({vertex, condensedVertices.size()});
            condensedVertices.push_back(vertex);
        }
        vertexIndexBuffer.at(i) = seenVerticesIndex.at(vertex);
    }

    for(unsigned int i = 0; i < condensedVertices.size(); i++) {
        fileContents << "v " << condensedVertices[i].x
                      << " " << condensedVertices[i].y
                      << " " << condensedVertices[i].z << std::endl;
    }

    fileContents << std::endl;

    if(hasNormalsEnabled) {
        for(unsigned int i = 0; i < mesh.vertexCount; i++) {
            ShapeDescriptor::cpu::float3 normal = mesh.normals[i];
            if(std::isnan(normal.x) || std::isnan(normal.z) || std::isnan(normal.z)) {
                normal = {1, 0, 0};
            }
            if(!seenNormalsIndex.contains(normal)) {
                seenNormalsIndex.insert({normal, condensedNormals.size()});
                condensedNormals.push_back(normal);
            }
            normalIndexBuffer.at(i) = seenNormalsIndex.at(normal);
        }

        for(unsigned int i = 0; i < condensedNormals.size(); i++) {
            fileContents << "vn " << condensedNormals[i].x
                         << " " << condensedNormals[i].y
                         << " " << condensedNormals[i].z << std::endl;
        }
    }

    if(useCustomTextureMap) {
        fileContents << std::endl;

        for(unsigned int i = 0; i < mesh.vertexCount; i++) {
            fileContents << "vt " << vertexTextureCoordinates.content[i].x << " " << vertexTextureCoordinates.content[i].y << std::endl;
        }
    }

    bool lastIterationWasHighlighted = false;

    fileContents << std::endl;

    for(unsigned int i = 0; i < mesh.vertexCount; i += 3) {
        bool currentIterationIsHighlighted = i >= highlightStartVertex && i < highlightEndVertex;

        if((hasHighlightsEnabled || useCustomTextureMap) && (currentIterationIsHighlighted != lastIterationWasHighlighted || i == 0)) {
            fileContents << std::endl;
            fileContents << "o object_" << i << std::endl;
            fileContents << "g object_" << i << std::endl;

            if(currentIterationIsHighlighted) {
                fileContents << "usemtl highlightMaterial" << std::endl;
            } else {
                fileContents << "usemtl regularMaterial" << std::endl;
            }
            fileContents << std::endl;
        }

        fileContents << "f "
           << (vertexIndexBuffer.at(i + 0) + 1) << "/"
           << (useCustomTextureMap ? std::to_string(i + 1) : "")
           << (hasNormalsEnabled ? "/" + std::to_string(normalIndexBuffer.at(i + 0) + 1) : "")
           << " "

           << (vertexIndexBuffer.at(i + 1) + 1) << "/"
           << (useCustomTextureMap ? std::to_string(i + 2) : "")
           << (hasNormalsEnabled ? "/" + std::to_string(normalIndexBuffer.at(i + 1) + 1) : "")
           << " "

           << (vertexIndexBuffer.at(i + 2) + 1) << "/"
           << (useCustomTextureMap ? std::to_string(i + 3) : "")
           << (hasNormalsEnabled ? "/" + std::to_string(normalIndexBuffer.at(i + 2) + 1) : "")
           << std::endl;

        lastIterationWasHighlighted = currentIterationIsHighlighted;
    }

    outputFile << fileContents.str();

    outputFile.close();
}

void ShapeDescriptor::writeOBJ(cpu::Mesh mesh, const std::filesystem::path outputFilePath) {
    // Highlight a range that will never be highlighted
    dumpMesh(mesh, outputFilePath, -1, -1, false, {0, nullptr}, "");
}

void ShapeDescriptor::writeOBJ(cpu::Mesh mesh, const std::filesystem::path outputFilePath,
        size_t highlightStartVertex, size_t highlightEndVertex) {

    dumpMesh(mesh, outputFilePath, highlightStartVertex, highlightEndVertex, false, {0, nullptr}, "");
}

void ShapeDescriptor::writeOBJ(cpu::Mesh mesh, const std::filesystem::path &outputFilePath,
          ShapeDescriptor::cpu::array<float2> vertexTextureCoordinates, std::string textureMapPath) {
    assert(vertexTextureCoordinates.length == mesh.vertexCount);
    dumpMesh(mesh, outputFilePath, -1, -1, true, vertexTextureCoordinates, textureMapPath);
}

