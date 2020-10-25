#pragma once

#include <shapeDescriptor/cpu/types/float3.h>
#include <vector>

#pragma GCC optimize ("0")
ShapeDescriptor::cpu::float3 hostComputeTriangleNormal(std::vector<ShapeDescriptor::cpu::float3> &vertices, unsigned int baseIndex);
ShapeDescriptor::cpu::float3 computeTriangleNormal(
        ShapeDescriptor::cpu::float3 &triangleVertex0,
        ShapeDescriptor::cpu::float3 &triangleVertex1,
        ShapeDescriptor::cpu::float3 &triangleVertex2);
#pragma GCC reset_options