#include <cmath>
#include "float3.h"
#include "float2.h"


float length(ShapeDescriptor::cpu::float3 vec) {
    return std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

ShapeDescriptor::cpu::float3 normalize(ShapeDescriptor::cpu::float3 in) {
    ShapeDescriptor::cpu::float3 out;
    float len = length(in);
    out.x = in.x / len;
    out.y = in.y / len;
    out.z = in.z / len;
    return out;
}

ShapeDescriptor::cpu::float3 operator* (ShapeDescriptor::cpu::float3 vec, float other) {
    ShapeDescriptor::cpu::float3 out;
    out.x = vec.x * other;
    out.y = vec.y * other;
    out.z = vec.z * other;
    return out;
}

ShapeDescriptor::cpu::float3 operator*(float other, ShapeDescriptor::cpu::float3 vec) {
    return operator*(vec, other);
}

std::ostream & operator<<(std::ostream & os, ShapeDescriptor::cpu::float3 vec) {
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return os;
}