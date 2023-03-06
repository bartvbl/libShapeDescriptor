#include <shapeDescriptor/utilities/kernels/gpuMeshSampler.cuh>
#include <shapeDescriptor/utilities/copy/pointCloud.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/free/pointCloud.h>
#include "meshSampler.h"

ShapeDescriptor::gpu::PointCloud ShapeDescriptor::utilities::sampleMesh(ShapeDescriptor::gpu::Mesh mesh, size_t sampleCount, size_t randomSamplingSeed) {
    return ShapeDescriptor::internal::sampleMesh(mesh, sampleCount, randomSamplingSeed);
}

ShapeDescriptor::cpu::PointCloud ShapeDescriptor::utilities::sampleMesh(ShapeDescriptor::cpu::Mesh mesh, size_t sampleCount, size_t randomSamplingSeed) {
    ShapeDescriptor::gpu::Mesh device_mesh = mesh.copyToGPU();
    ShapeDescriptor::gpu::PointCloud device_pointCloud = sampleMesh(device_mesh, sampleCount, randomSamplingSeed);
    ShapeDescriptor::cpu::PointCloud pointCloud = ShapeDescriptor::copy::devicePointCloudToHost(device_pointCloud);
    ShapeDescriptor::free::mesh(device_mesh);
    ShapeDescriptor::free::pointCloud(device_pointCloud);
    return pointCloud;
}


