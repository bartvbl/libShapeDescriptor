#include "spinOriginsGenerator.h"

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
    return ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::OrientedPoint>();
}
