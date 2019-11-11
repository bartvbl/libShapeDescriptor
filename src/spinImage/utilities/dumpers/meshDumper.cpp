#include "meshDumper.h"
#include <fstream>
#include <iostream>

void SpinImage::dump::mesh(cpu::Mesh mesh, std::string outputFilePath) {
    std::ofstream outputFile;
    outputFile.open(outputFilePath);

    for(unsigned int i = 0; i < mesh.vertexCount; i++) {
        outputFile << "v " << mesh.vertices[i].x << " " << mesh.vertices[i].y << " " <<mesh.vertices[i].z << std::endl;
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