#include "meshDumper.h"
#include <fstream>
#include <iostream>
#include <vector>

void dumpMesh(SpinImage::cpu::Mesh mesh, std::string &outputFilePath, std::vector<SpinImage::cpu::float2> &textureCoordinates) {

    bool hasHighlightsEnabled = textureCoordinates.size() > 0;

    if(hasHighlightsEnabled) {
        std::ofstream materialFile;
        materialFile.open();

        materialFile << "newmtl highlightMaterial" << std::endl;
        materialFile << "Ns 10.0000" << std::endl;
        materialFile << "Ni 1.5000" << std::endl;
        materialFile << "d 1.0000" << std::endl;
        materialFile << "Tr 0.0000" << std::endl;
        materialFile << "Tf 1.0000 1.0000 1.0000" << std::endl;
        materialFile << "illum 2" << std::endl;
        materialFile << "Ka 0.0000 0.0000 0.0000" << std::endl;
        materialFile << "Kd 1.0000 1.0000 1.0000" << std::endl;
        materialFile << "Ks 1.0000 1.0000 1.0000" << std::endl;
        materialFile << "Ke 0.0000 0.0000 0.0000" << std::endl;
    }

    std::ofstream outputFile;
    outputFile.open(outputFilePath);

    outputFile << "mtlllib exportedObject.mtl" << std::endl;

    outputFile << std::endl;

    for(unsigned int i = 0; i < mesh.vertexCount; i++) {
        outputFile << "v " << mesh.vertices[i].x << " " << mesh.vertices[i].y << " " <<mesh.vertices[i].z << std::endl;
    }

    outputFile << std::endl;

    if(hasHighlightsEnabled) {
        for(unsigned int i = 0; i < mesh.vertexCount; i++) {
            outputFile << "vt " << textureCoordinates[i].x << " " << textureCoordinates[i].y << std::endl;
        }

        outputFile << std::endl;
    }

    for(unsigned int i = 0; i < mesh.vertexCount; i++) {
        outputFile << "vn " << mesh.normals[i].x << " " << mesh.normals[i].y << " " <<mesh.normals[i].z << std::endl;
    }

    outputFile << std::endl;
    outputFile << "o object" << std::endl;
    outputFile << "g object" << std::endl;

    if(hasHighlightsEnabled) {
        outputFile << "usemtl highlightMaterial" << std::endl;
    }

    outputFile << std::endl;

    for(unsigned int i = 0; i < mesh.vertexCount; i += 3) {
        if(hasHighlightsEnabled) {
            outputFile << "f "
                       << (i+1) << "//" << (i+1) << " "
                       << (i+2) << "//" << (i+2) << " "
                       << (i+3) << "//" << (i+3) << std::endl;
        } else {
            outputFile << "f "
                       << (i+1) << "/" << (i+1) << "/" << (i+1) << " "
                       << (i+2) << "/" << (i+2) << "/" << (i+2) << " "
                       << (i+3) << "/" << (i+3) << "/" << (i+3) << std::endl;
        }
    }

    outputFile.close();
}

void SpinImage::dump::mesh(cpu::Mesh mesh, std::string outputFilePath) {
    std::vector<SpinImage::cpu::float2> empty;
    dumpMesh(mesh, outputFilePath, empty);
}

void SpinImage::dump::mesh(cpu::Mesh mesh, std::string outputFilePath,
        size_t highlightStartVertex, size_t highlightEndVertex) {
    std::vector<SpinImage::cpu::float2> textureCoordinates;
    textureCoordinates.resize(mesh.vertexCount);

    for(size_t i = 0; i < mesh.vertexCount; i++) {
        if(i >= highlightStartVertex && i < highlightEndVertex) {
            textureCoordinates.at(i).x = 1;
            textureCoordinates.at(i).y = 1;
        } else {
            textureCoordinates.at(i).x = 0;
            textureCoordinates.at(i).y = 0;
        }
    }

    dumpMesh(mesh, outputFilePath, textureCoordinates);
}

