#include <iostream>
#include "MeshLoader.h"
#include "PLYLoader.h"
#include "OBJLoader.h"
#include "OFFLoader.h"
#include "GLTFLoader.h"

ShapeDescriptor::cpu::Mesh
ShapeDescriptor::utilities::loadMesh(std::filesystem::path src, RecomputeNormals recomputeNormals) {
    if(!std::filesystem::exists(src)) {
        throw std::runtime_error("The file \"" + src.string() + "\" was not found, and could therefore not be loaded.");
    }
    if(src.extension() == ".ply" || src.extension() == ".PLY") {
        return ShapeDescriptor::utilities::loadPLY(src.string(), recomputeNormals);
    } else if(src.extension() == ".obj" || src.extension() == ".OBJ") {
        return ShapeDescriptor::utilities::loadOBJ(src.string(), recomputeNormals);
    } else if(src.extension() == ".off" || src.extension() == ".OFF") {
        return ShapeDescriptor::utilities::loadOFF(src.string());
    } else if(src.extension() == ".gltf" || src.extension() == ".GLTF" || src.extension() == ".glb" || src.extension() == ".GLB") {
        return ShapeDescriptor::utilities::loadGLTFMesh(src, recomputeNormals);
    } else {
        throw std::runtime_error("Failed to load file: " + src.string() + "\nReason: extension was not recognised as a supported 3D object file format.");
    }

}
