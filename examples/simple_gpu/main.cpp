#include <shapeDescriptor/shapeDescriptor.h>
#include <iostream>

int main(int argc, char** argv) {
    if(argc == 1) {
        std::cout << "Usage: simple_gpu [file_to_read.obj/.ply/.off]" << std::endl;
        return 1;
    }

    // Load mesh
    std::string fileToRead = std::string(argv[1]);
    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::loadMesh(fileToRead, ShapeDescriptor::RecomputeNormals::RECOMPUTE_IF_MISSING);
        
    // Store it on the GPU
    ShapeDescriptor::gpu::Mesh gpuMesh = ShapeDescriptor::copyToGPU(mesh);

    // Define and upload descriptor origins
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins = ShapeDescriptor::generateUniqueSpinOriginBuffer(mesh);

    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> gpuDescriptorOrigins = descriptorOrigins.copyToGPU();

    // Compute the descriptor(s)
    float supportRadius = 1.0;
    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> descriptors = 
        ShapeDescriptor::generateRadialIntersectionCountImages(
                gpuMesh,
                gpuDescriptorOrigins,
                supportRadius);
                
    // Copy descriptors to RAM
    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> hostDescriptors = descriptors.copyToCPU();
                    
    // Do something with descriptors here, for example write the first 5000 to an image file
    descriptors.length = std::min<size_t>(hostDescriptors.length, 5000);
    ShapeDescriptor::writeDescriptorImages(hostDescriptors, "output_image.png");

    // Free memory
    ShapeDescriptor::free(descriptorOrigins);
    ShapeDescriptor::free(hostDescriptors);
    ShapeDescriptor::free(gpuDescriptorOrigins);
    ShapeDescriptor::free(descriptors);
    ShapeDescriptor::free(mesh);
    ShapeDescriptor::free(gpuMesh);
}