#include <shapeDescriptor/utilities/kernels/gpuMeshSampler.cuh>
#include "meshSampler.h"

ShapeDescriptor::gpu::PointCloud ShapeDescriptor::utilities::sampleMesh(ShapeDescriptor::gpu::Mesh mesh, size_t sampleCount, size_t randomSamplingSeed) {
    return ShapeDescriptor::internal::sampleMesh(mesh, sampleCount, randomSamplingSeed);
}

ShapeDescriptor::cpu::PointCloud ShapeDescriptor::utilities::sampleMesh(ShapeDescriptor::cpu::Mesh mesh, size_t sampleCount, size_t randomSamplingSeed) {
    return ShapeDescriptor::cpu::PointCloud();
}


