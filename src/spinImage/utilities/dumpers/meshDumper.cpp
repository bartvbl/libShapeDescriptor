#include "meshDumper.h"
#include <fstream>
#include <iostream>
#include <vector>

void dumpMesh(SpinImage::cpu::Mesh mesh, std::experimental::filesystem::path &outputFilePath, size_t highlightStartVertex, size_t highlightEndVertex) {

    bool hasHighlightsEnabled =
            (highlightStartVertex > 0 && highlightStartVertex < mesh.vertexCount) ||
            (highlightEndVertex > 0 && highlightEndVertex < mesh.vertexCount);

    std::ofstream outputFile;
    outputFile.open(outputFilePath);

    if(hasHighlightsEnabled) {
        std::experimental::filesystem::path materialLibPath =
                outputFilePath.parent_path() / "exportedObject.mtl";

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
        materialFile << "Kd 0.8000 0.8000 0.8000" << std::endl;
        materialFile << "Ks 0.8000 0.8000 0.8000" << std::endl;
        materialFile << "Ke 0.0000 0.0000 0.0000" << std::endl << std::endl;

        materialFile << "newmtl highlightMaterial" << std::endl;
        materialFile << "Ns 10.0000" << std::endl;
        materialFile << "Ni 1.5000" << std::endl;
        materialFile << "d 1.0000" << std::endl;
        materialFile << "Tr 0.0000" << std::endl;
        materialFile << "Tf 1.0000 1.0000 1.0000" << std::endl;
        materialFile << "illum 2" << std::endl;
        materialFile << "Ka 0.0000 0.0000 0.0000" << std::endl;
        materialFile << "Kd 0.6000 0.0000 0.0000" << std::endl;
        materialFile << "Ks 0.8000 0.8000 0.8000" << std::endl;
        materialFile << "Ke 0.0000 0.0000 0.0000" << std::endl;

        materialFile.close();

        outputFile << "mtllib exportedObject.mtl" << std::endl;

        outputFile << std::endl;
    }

    for(unsigned int i = 0; i < mesh.vertexCount; i++) {
        outputFile << "v " << mesh.vertices[i].x << " " << mesh.vertices[i].y << " " <<mesh.vertices[i].z << std::endl;
    }

    outputFile << std::endl;

    for(unsigned int i = 0; i < mesh.vertexCount; i++) {
        outputFile << "vn " << mesh.normals[i].x << " " << mesh.normals[i].y << " " <<mesh.normals[i].z << std::endl;
    }

    bool lastIterationWasHighlighted = false;

    outputFile << std::endl;

    for(unsigned int i = 0; i < mesh.vertexCount; i += 3) {
        bool currentIterationIsHighlighted = i >= highlightStartVertex && i < highlightEndVertex;

        if(currentIterationIsHighlighted != lastIterationWasHighlighted || i == 0) {
            outputFile << std::endl;
            outputFile << "o object_" << i << std::endl;
            outputFile << "g object_" << i << std::endl;

            if(currentIterationIsHighlighted) {
                outputFile << "usemtl highlightMaterial" << std::endl;
            } else {
                outputFile << "usemtl regularMaterial" << std::endl;
            }
            outputFile << std::endl;
        }

        outputFile << "f "
           << (i+1) << "//" << (i+1) << " "
           << (i+2) << "//" << (i+2) << " "
           << (i+3) << "//" << (i+3) << std::endl;
        lastIterationWasHighlighted = currentIterationIsHighlighted;
    }

    outputFile.close();
}

void SpinImage::dump::mesh(cpu::Mesh mesh, std::experimental::filesystem::path outputFilePath) {
    // Highlight a range that will never be highlighted
    dumpMesh(mesh, outputFilePath, -1, -1);
}

void SpinImage::dump::mesh(cpu::Mesh mesh, std::experimental::filesystem::path outputFilePath,
        size_t highlightStartVertex, size_t highlightEndVertex) {

    dumpMesh(mesh, outputFilePath, highlightStartVertex, highlightEndVertex);
}

