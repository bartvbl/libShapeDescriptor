#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>

ShapeDescriptor::cpu::PointCloud
ShapeDescriptor::loadPointCloud(std::filesystem::path src) {
    if(src.extension() == ".ply" || src.extension() == ".PLY") {
        ShapeDescriptor::cpu::Mesh loadedMesh = ShapeDescriptor::loadPLY(src.string(), RecomputeNormals::DO_NOT_RECOMPUTE);
        ShapeDescriptor::cpu::PointCloud cloud(loadedMesh.vertexCount);
        cloud.vertices = loadedMesh.vertices;
        cloud.normals = loadedMesh.normals;
        cloud.vertexColours = loadedMesh.vertexColours;
        cloud.hasVertexNormals = loadedMesh.normals != nullptr;
        cloud.hasVertexColours = loadedMesh.vertexColours != nullptr;
        return cloud;
    } else if(src.extension() == ".xyz" || src.extension() == ".XYZ") {
        return ShapeDescriptor::loadXYZ(src);
    } else if(src.extension() == ".xyzn" || src.extension() == ".XYZN") {
        return ShapeDescriptor::loadXYZ(src, true);
    } else if(src.extension() == ".xyzrgb" || src.extension() == ".XYZRGB") {
        return ShapeDescriptor::loadXYZ(src, false, true);
    } else if(src.extension() == ".glb" || src.extension() == ".GLB" || src.extension() == ".gltf" || src.extension() == ".GLTF") {
        return ShapeDescriptor::loadGLTFPointCloud(src);
    } else {
        throw std::runtime_error("Failed to load point cloud file: " + src.string() + "\nReason: extension was not recognised as a supported 3D object file format.");
    }
}
