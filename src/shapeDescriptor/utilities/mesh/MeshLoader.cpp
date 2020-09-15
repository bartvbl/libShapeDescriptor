#include <iostream>
#include "MeshLoader.h"
#include "PLYLoader.h"
#include "OBJLoader.h"
#include "OFFLoader.h"

ShapeDescriptor::cpu::Mesh
ShapeDescriptor::utilities::loadMesh(std::experimental::filesystem::path src, bool recomputeNormals) {
    if(src.extension() == ".ply" || src.extension() == ".PLY") {
        return ShapeDescriptor::utilities::loadPLY(src, recomputeNormals);
    } else if(src.extension() == ".obj" || src.extension() == ".OBJ") {
        return ShapeDescriptor::utilities::loadOBJ(src, recomputeNormals);
    } else if(src.extension() == ".off" || src.extension() == ".OFF") {
        return ShapeDescriptor::utilities::loadOFF(src);
    } else {
        throw std::runtime_error("Failed to load file: " + src.string() + "\nReason: extension was not recognised as a supported 3D object file format.");
    }
}
