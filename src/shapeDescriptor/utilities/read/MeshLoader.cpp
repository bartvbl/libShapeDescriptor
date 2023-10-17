#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>
ShapeDescriptor::cpu::Mesh
ShapeDescriptor::loadMesh(std::filesystem::path src, RecomputeNormals recomputeNormals) {
    if(!std::filesystem::exists(src)) {
        throw std::runtime_error("The file \"" + src.string() + "\" was not found, and could therefore not be loaded.");
    }
    if(src.extension() == ".ply" || src.extension() == ".PLY") {
        return ShapeDescriptor::loadPLY(src.string(), recomputeNormals);
    } else if(src.extension() == ".obj" || src.extension() == ".OBJ") {
        return ShapeDescriptor::loadOBJ(src.string(), recomputeNormals);
    } else if(src.extension() == ".off" || src.extension() == ".OFF") {
        return ShapeDescriptor::loadOFF(src.string());
    } else if(src.extension() == ".gltf" || src.extension() == ".GLTF" || src.extension() == ".glb" || src.extension() == ".GLB") {
        return ShapeDescriptor::loadGLTFMesh(src, recomputeNormals);
    } else if(src.extension() == ".cm" || src.extension() == ".CM") {
        return ShapeDescriptor::loadMeshFromCompressedGeometryFile(src);
    } else {
        throw std::runtime_error("Failed to load file: " + src.string() + "\nReason: extension was not recognised as a supported 3D object file format.");
    }

}
