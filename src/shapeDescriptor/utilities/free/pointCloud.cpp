#include <shapeDescriptor/utilities/free/pointCloud.h>

void ShapeDescriptor::free::pointCloud(ShapeDescriptor::cpu::PointCloud &cloudToFree) {
    if(cloudToFree.vertices != nullptr) {
        delete[] cloudToFree.vertices;
        cloudToFree.vertices = nullptr;
    }

    if(cloudToFree.normals != nullptr) {
        delete[] cloudToFree.normals;
        cloudToFree.normals = nullptr;
    }

    if(cloudToFree.vertexColours != nullptr) {
        delete[] cloudToFree.vertexColours;
        cloudToFree.vertexColours = nullptr;
    }
}

void ShapeDescriptor::free::pointCloud(ShapeDescriptor::gpu::PointCloud &cloudToFree) {
    cloudToFree.free();
}