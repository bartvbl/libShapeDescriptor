#include "meshDumper.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>

void dumpMesh(ShapeDescriptor::cpu::Mesh mesh, const std::experimental::filesystem::path &outputFilePath, size_t highlightStartVertex, size_t highlightEndVertex,
        bool useCustomTextureMap, ShapeDescriptor::cpu::array<float2> vertexTextureCoordinates, std::string textureMapPath) {

    bool hasHighlightsEnabled =
            (highlightStartVertex > 0 && highlightStartVertex <= mesh.vertexCount) ||
            (highlightEndVertex > 0 && highlightEndVertex <= mesh.vertexCount);

    std::ofstream outputFile;
    outputFile.open(outputFilePath);
    const std::string materialLibFileName = useCustomTextureMap ? "highlightObject.mtl" : "exportedObject.mtl";

    std::stringstream fileContents;

    if(hasHighlightsEnabled || useCustomTextureMap) {
        std::experimental::filesystem::path materialLibPath =
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

    for(unsigned int i = 0; i < mesh.vertexCount; i++) {
        fileContents << "v " << mesh.vertices[i].x << " " << mesh.vertices[i].y << " " <<mesh.vertices[i].z << std::endl;
    }

    fileContents << std::endl;

    for(unsigned int i = 0; i < mesh.vertexCount; i++) {
        fileContents << "vn " << mesh.normals[i].x << " " << mesh.normals[i].y << " " <<mesh.normals[i].z << std::endl;
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
           << (i+1) << "/" << (i+1) << "/" << (i+1) << " "
           << (i+2) << "/" << (i+2) << "/" << (i+2) << " "
           << (i+3) << "/" << (i+3) << "/" << (i+3) << std::endl;
        lastIterationWasHighlighted = currentIterationIsHighlighted;
    }

    outputFile << fileContents.str();

    outputFile.close();
}

void ShapeDescriptor::dump::mesh(cpu::Mesh mesh, const std::experimental::filesystem::path outputFilePath) {
    // Highlight a range that will never be highlighted
    dumpMesh(mesh, outputFilePath, -1, -1, false, {0, nullptr}, "");
}

void ShapeDescriptor::dump::mesh(cpu::Mesh mesh, const std::experimental::filesystem::path outputFilePath,
        size_t highlightStartVertex, size_t highlightEndVertex) {

    dumpMesh(mesh, outputFilePath, highlightStartVertex, highlightEndVertex, false, {0, nullptr}, "");
}

void ShapeDescriptor::dump::mesh(cpu::Mesh mesh, const std::experimental::filesystem::path &outputFilePath,
          ShapeDescriptor::cpu::array<float2> vertexTextureCoordinates, std::string textureMapPath) {
    assert(vertexTextureCoordinates.length == mesh.vertexCount);
    dumpMesh(mesh, outputFilePath, -1, -1, true, vertexTextureCoordinates, textureMapPath);
}

