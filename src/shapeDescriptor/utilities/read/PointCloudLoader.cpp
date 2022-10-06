#include <iostream>
#include "PointCloudLoader.h"
#include "PLYLoader.h"

ShapeDescriptor::cpu::PointCloud
ShapeDescriptor::utilities::loadPointCloud(std::filesystem::path src, bool recomputeNormals) {
    if(src.extension() == ".ply" || src.extension() == ".PLY") {
        ShapeDescriptor::cpu::Mesh loadedMesh = ShapeDescriptor::utilities::loadPLY(src.string(), recomputeNormals);
        ShapeDescriptor::cpu::PointCloud cloud(loadedMesh.vertexCount);
        cloud.vertices = loadedMesh.vertices;
        cloud.normals = loadedMesh.normals;
        cloud.vertexColours = loadedMesh.vertexColours;
        return cloud;
    } else {
        throw std::runtime_error("Failed to load file: " + src.string() + "\nReason: extension was not recognised as a supported 3D object file format.");
    }
}
