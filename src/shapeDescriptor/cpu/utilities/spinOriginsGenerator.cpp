#include "spinOriginsGenerator.h"
#include <unordered_set>
#include <vector>

ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::OrientedPoint>
ShapeDescriptor::cpu::generateSpinOriginBuffer(ShapeDescriptor::cpu::Mesh &mesh) {
    ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::OrientedPoint> originBuffer(mesh.vertexCount);
    for(size_t i = 0; i < mesh.vertexCount; i++) {
        originBuffer.content[i] = ShapeDescriptor::cpu::OrientedPoint{mesh.vertices[i], mesh.normals[i]};
    }
    return originBuffer;
}

ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::OrientedPoint>
ShapeDescriptor::cpu::generateUniqueSpinOriginBuffer(ShapeDescriptor::cpu::Mesh &mesh) {
    std::vector<ShapeDescriptor::cpu::OrientedPoint> originBuffer;
    originBuffer.reserve(mesh.vertexCount);
    std::unordered_set<ShapeDescriptor::cpu::OrientedPoint> seenSet;
    for(size_t i = 0; i < mesh.vertexCount; i++) {
        ShapeDescriptor::cpu::OrientedPoint currentPoint = ShapeDescriptor::cpu::OrientedPoint{mesh.vertices[i], mesh.normals[i]};
        if(seenSet.count(currentPoint) == 0) {
            seenSet.insert(currentPoint);
            originBuffer.emplace_back(currentPoint);
        }
    }

    ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::OrientedPoint> outputBuffer(originBuffer.size());
    std::copy(originBuffer.begin(), originBuffer.end(), outputBuffer.content);
    return outputBuffer;
}
