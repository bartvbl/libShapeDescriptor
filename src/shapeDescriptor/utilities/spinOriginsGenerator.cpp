#include "spinOriginsGenerator.h"
#include <unordered_set>
#include <vector>

ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>
ShapeDescriptor::utilities::generateSpinOriginBuffer(ShapeDescriptor::cpu::Mesh &mesh) {
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> originBuffer(mesh.vertexCount);
    for(size_t i = 0; i < mesh.vertexCount; i++) {
        originBuffer.content[i] = ShapeDescriptor::OrientedPoint{mesh.vertices[i], mesh.normals[i]};
    }
    return originBuffer;
}

ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>
ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(ShapeDescriptor::cpu::Mesh &mesh) {
    std::vector<ShapeDescriptor::OrientedPoint> originBuffer;
    originBuffer.reserve(mesh.vertexCount);
    std::unordered_set<ShapeDescriptor::OrientedPoint> seenSet;
    for(size_t i = 0; i < mesh.vertexCount; i++) {
        ShapeDescriptor::OrientedPoint currentPoint = ShapeDescriptor::OrientedPoint{mesh.vertices[i], mesh.normals[i]};
        if(seenSet.count(currentPoint) == 0) {
            seenSet.insert(currentPoint);
            originBuffer.emplace_back(currentPoint);
        }
    }

    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> outputBuffer(originBuffer.size());
    std::copy(originBuffer.begin(), originBuffer.end(), outputBuffer.content);
    return outputBuffer;
}
