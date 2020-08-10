#include <cmath>
#include "float2.h"

float length(ShapeDescriptor::cpu::float2 vec) {
    return std::sqrt(vec.x * vec.x + vec.y * vec.y);
}

ShapeDescriptor::cpu::float2 normalize(ShapeDescriptor::cpu::float2 in) {
    ShapeDescriptor::cpu::float2 out;
    float len = length(in);
    out.x = in.x / len;
    out.y = in.y / len;
    return out;
}