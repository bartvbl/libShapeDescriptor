#include "meshDumper.h"
#include <fstream>
#include <iostream>
#include <vector>

void dumpMesh(SpinImage::cpu::Mesh mesh, std::string &outputFilePath, std::vector<SpinImage::cpu::float2> &textureCoordinates) {

    std::ofstream materialFile;


    std::ofstream outputFile;
    outputFile.open(outputFilePath);

    outputFile << "mtlllib exportedObject.mtl" << std::endl;
    
    outputFile << std::endl;

    for(unsigned int i = 0; i < mesh.vertexCount; i++) {
        outputFile << "v " << mesh.vertices[i].x << " " << mesh.vertices[i].y << " " <<mesh.vertices[i].z << std::endl;
    }

    outputFile << std::endl;

    for(unsigned int i = 0; i < mesh.vertexCount; i++) {
        outputFile << "vt " << textureCoordinates[i].x << " " << textureCoordinates[i].y << std::endl;
    }

    outputFile << std::endl;

    for(unsigned int i = 0; i < mesh.vertexCount; i++) {
        outputFile << "vn " << mesh.normals[i].x << " " << mesh.normals[i].y << " " <<mesh.normals[i].z << std::endl;
    }

    outputFile << std::endl;
    outputFile << "o object" << std::endl;
    outputFile << "g object" << std::endl;
    outputFile << std::endl;

    for(unsigned int i = 0; i < mesh.vertexCount; i += 3) {
        outputFile << "f " << (i+1) << "//" << (i+1) << " " << (i+2) << "//" << (i+2) << " " << (i+3) << "//" << (i+3) << std::endl;
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

