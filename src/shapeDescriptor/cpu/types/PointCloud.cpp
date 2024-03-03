#include <shapeDescriptor/shapeDescriptor.h>

ShapeDescriptor::cpu::PointCloud ShapeDescriptor::cpu::PointCloud::clone() const {
    ShapeDescriptor::cpu::PointCloud copiedCloud(pointCount);

    std::copy(vertices, vertices + pointCount, copiedCloud.vertices);

    if(normals != nullptr) {
        std::copy(normals, normals + pointCount, copiedCloud.normals);
    } else {
        delete[] copiedCloud.normals;
        copiedCloud.normals = nullptr;
    }

    if(vertexColours != nullptr) {
        assert(copiedCloud.vertexColours != nullptr);
        std::copy(vertexColours, vertexColours + pointCount, copiedCloud.vertexColours);
    } else {
        delete[] copiedCloud.vertexColours;
        copiedCloud.vertexColours = nullptr;
    }

    return copiedCloud;
}