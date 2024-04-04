#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>

void ShapeDescriptor::writePointCloud(const ShapeDescriptor::cpu::PointCloud& cloud, std::filesystem::path destination) {
    if(destination.extension() == ".xyz" || destination.extension() == ".XYZ") {
        ShapeDescriptor::writeXYZ(destination.string(), cloud);
    } else {
        throw std::runtime_error("Failed to write file: " + destination.string() + "\nReason: extension was not recognised as a supported 3D object file format.");
    }
}