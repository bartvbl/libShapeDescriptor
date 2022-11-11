#include <shapeDescriptor/common/types/OrientedPoint.h>
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/utilities/read/MeshLoader.h>
#include <shapeDescriptor/utilities/copy/array.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/dump/descriptorImages.h>

int main(int argc, char** argv) {
    if(argc == 1) {
        std::cout << "Usage: simple_gpu [file_to_read.obj/.ply/.off]" << std::endl;
        return 1;
    }

    // Load mesh
    const bool recomputeNormals = false;
    std::string fileToRead = std::string(argv[1]);
    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadMesh(fileToRead, recomputeNormals);
        
    // Store it on the GPU
    ShapeDescriptor::gpu::Mesh gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);

    // Define and upload descriptor origins
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(mesh);

    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> gpuDescriptorOrigins = 
        ShapeDescriptor::copy::hostArrayToDevice(descriptorOrigins);

    // Compute the descriptor(s)
    float supportRadius = 1.0;
    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> descriptors = 
        ShapeDescriptor::gpu::generateRadialIntersectionCountImages(
                gpuMesh,
                gpuDescriptorOrigins,
                supportRadius);
                
    // Copy descriptors to RAM
    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> hostDescriptors =
                ShapeDescriptor::copy::deviceArrayToHost(descriptors);
                    
    // Do something with descriptors here, for example write the first 5000 to an image file
    descriptors.length = std::min<size_t>(hostDescriptors.length, 5000);
    ShapeDescriptor::dump::descriptors(hostDescriptors, "output_image.png");

    // Free memory
    ShapeDescriptor::free::array(descriptorOrigins);
    ShapeDescriptor::free::array(hostDescriptors);
    ShapeDescriptor::free::array(gpuDescriptorOrigins);
    ShapeDescriptor::free::array(descriptors);
    ShapeDescriptor::free::mesh(mesh);
    ShapeDescriptor::free::mesh(gpuMesh);
}