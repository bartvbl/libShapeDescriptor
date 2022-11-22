#include <iostream>
#include "PointCloudLoader.h"
#include "PLYLoader.h"
#include "XYZLoader.h"

ShapeDescriptor::cpu::PointCloud
ShapeDescriptor::utilities::loadPointCloud(std::filesystem::path src) {
    if(src.extension() == ".ply" || src.extension() == ".PLY") {
        ShapeDescriptor::cpu::Mesh loadedMesh = ShapeDescriptor::utilities::loadPLY(src.string(), false);
        ShapeDescriptor::cpu::PointCloud cloud(loadedMesh.vertexCount);
        cloud.vertices = loadedMesh.vertices;
        cloud.normals = loadedMesh.normals;
        cloud.vertexColours = loadedMesh.vertexColours;
        cloud.hasVertexNormals = loadedMesh.normals != nullptr;
        cloud.hasVertexColours = loadedMesh.vertexColours != nullptr;
        return cloud;
    } else if(src.extension() == ".xyz" || src.extension() == ".XYZ") {
        return ShapeDescriptor::utilities::loadXYZ(src);
    } else if(src.extension() == ".xyzn" || src.extension() == ".XYZN") {
        return ShapeDescriptor::utilities::loadXYZ(src, true);
    } else if(src.extension() == ".xyzrgb" || src.extension() == ".XYZRGB") {
        return ShapeDescriptor::utilities::loadXYZ(src, false, true);
    } else {
        throw std::runtime_error("Failed to load point cloud file: " + src.string() + "\nReason: extension was not recognised as a supported 3D object file format.");
    }
}
