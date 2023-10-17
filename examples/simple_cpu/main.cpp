#include <shapeDescriptor/shapeDescriptor.h>
#include <iostream>

int main(int argc, char** argv) {
    if(argc == 1) {
        std::cout << "Usage: simple_cpu [file_to_read.obj/.ply/.off]" << std::endl;
        return 1;
    }

    // Load mesh
    std::string fileToRead = std::string(argv[1]);
    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::loadMesh(fileToRead, ShapeDescriptor::RecomputeNormals::RECOMPUTE_IF_MISSING);
        
    // Define and upload descriptor origins
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins = ShapeDescriptor::generateUniqueSpinOriginBuffer(mesh);

    // Compute the descriptor(s)
    float supportRadius = 1.0;
    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptors = 
        ShapeDescriptor::generateRadialIntersectionCountImages(
                mesh,
                descriptorOrigins,
                supportRadius);

                    
    // Do something with descriptors here, for example write the first 5000 to an image file
    descriptors.length = std::min<size_t>(descriptors.length, 5000);
    ShapeDescriptor::writeDescriptorImages(descriptors, "output_image.png");

    // Free memory
    ShapeDescriptor::free(descriptorOrigins);
    ShapeDescriptor::free(descriptors);
    ShapeDescriptor::free(mesh);
}