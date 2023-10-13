#include <shapeDescriptor/shapeDescriptor.h>

ShapeDescriptor::gpu::PointCloud ShapeDescriptor::sampleMesh(ShapeDescriptor::gpu::Mesh mesh, size_t sampleCount, size_t randomSamplingSeed, internal::MeshSamplingBuffers* keepComputedBuffersForExternalUse) {
    return ShapeDescriptor::sampleMesh(mesh, sampleCount, randomSamplingSeed);
}

ShapeDescriptor::cpu::PointCloud ShapeDescriptor::sampleMesh(ShapeDescriptor::cpu::Mesh mesh, size_t sampleCount, size_t randomSamplingSeed) {
    ShapeDescriptor::gpu::Mesh device_mesh = mesh.copyToGPU();
    ShapeDescriptor::gpu::PointCloud device_pointCloud = sampleMesh(device_mesh, sampleCount, randomSamplingSeed);
    ShapeDescriptor::cpu::PointCloud pointCloud = ShapeDescriptor::copyToCPU(device_pointCloud);
    ShapeDescriptor::free(device_mesh);
    ShapeDescriptor::free(device_pointCloud);
    return pointCloud;
}


