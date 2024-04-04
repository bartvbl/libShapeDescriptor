#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>

void ShapeDescriptor::writeMesh(const ShapeDescriptor::cpu::Mesh& mesh, std::filesystem::path destination) {
    if(destination.extension() == ".obj" || destination.extension() == ".OBJ") {
        ShapeDescriptor::writeOBJ(mesh, destination);
    } else {
        throw std::runtime_error("Failed to write file: " + destination.string() + "\nReason: extension was not recognised as a supported 3D object file format.");
    }
}