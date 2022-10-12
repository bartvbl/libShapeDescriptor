#include "pointCloud.h"
#include "VertexList.h"

ShapeDescriptor::gpu::PointCloud ShapeDescriptor::copy::hostPointCloudToDevice(cpu::PointCloud hostCloud) {
    gpu::PointCloud device_cloud;
    device_cloud.vertices = ShapeDescriptor::copy::hostVertexListToDevice({hostCloud.pointCount, hostCloud.vertices});
    if(hostCloud.normals != nullptr) {
        device_cloud.normals = ShapeDescriptor::copy::hostVertexListToDevice({hostCloud.pointCount, hostCloud.normals});
    }
    device_cloud.pointCount = hostCloud.pointCount;

    return device_cloud;
}

ShapeDescriptor::cpu::PointCloud ShapeDescriptor::copy::devicePointCloudToHost(gpu::PointCloud deviceCloud) {
    cpu::PointCloud hostCloud;
    hostCloud.vertices = ShapeDescriptor::copy::deviceVertexListToHost(deviceCloud.vertices).content;
    if(deviceCloud.normals.length > 0) {
        hostCloud.normals = ShapeDescriptor::copy::deviceVertexListToHost(deviceCloud.normals).content;
    }
    hostCloud.vertexColours = nullptr;
    hostCloud.pointCount = deviceCloud.pointCount;
    return hostCloud;
}