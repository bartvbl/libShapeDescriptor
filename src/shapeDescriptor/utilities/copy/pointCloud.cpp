#include <shapeDescriptor/shapeDescriptor.h>

ShapeDescriptor::gpu::PointCloud ShapeDescriptor::copyToGPU(cpu::PointCloud hostCloud) {
    gpu::PointCloud device_cloud;
    device_cloud.vertices = ShapeDescriptor::copyToGPU({hostCloud.pointCount, hostCloud.vertices});
    if(hostCloud.normals != nullptr) {
        device_cloud.normals = ShapeDescriptor::copyToGPU({hostCloud.pointCount, hostCloud.normals});
    }
    device_cloud.pointCount = hostCloud.pointCount;

    return device_cloud;
}

ShapeDescriptor::cpu::PointCloud ShapeDescriptor::copyToCPU(gpu::PointCloud deviceCloud) {
    cpu::PointCloud hostCloud;
    hostCloud.vertices = ShapeDescriptor::copyToCPU(deviceCloud.vertices).content;
    if(deviceCloud.normals.length > 0) {
        hostCloud.normals = ShapeDescriptor::copyToCPU(deviceCloud.normals).content;
    }
    hostCloud.vertexColours = nullptr;
    hostCloud.pointCount = deviceCloud.pointCount;
    return hostCloud;
}