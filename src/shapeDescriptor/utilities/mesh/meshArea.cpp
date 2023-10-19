#include <shapeDescriptor/shapeDescriptor.h>

double ShapeDescriptor::calculateMeshSurfaceArea(const cpu::Mesh &mesh) {
    double totalArea = 0;

    for(uint32_t i = 0; i < mesh.vertexCount; i += 3) {
        ShapeDescriptor::cpu::float3 vertex0 = mesh.vertices[i];
        ShapeDescriptor::cpu::float3 vertex1 = mesh.vertices[i + 1];
        ShapeDescriptor::cpu::float3 vertex2 = mesh.vertices[i + 2];

        ShapeDescriptor::cpu::float3 AB = vertex1 - vertex0;
        ShapeDescriptor::cpu::float3 AC = vertex2 - vertex0;

        totalArea += length(cross(AB, AC)) * 0.5;
    }

    return totalArea;
}