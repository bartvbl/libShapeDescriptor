#include <cmath>
#include "float3.h"
#include "float2.h"


float length(SpinImage::cpu::float3 vec) {
    return std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

SpinImage::cpu::float3 normalize(SpinImage::cpu::float3 in) {
    SpinImage::cpu::float3 out;
    float len = length(in);
    out.x = in.x / len;
    out.y = in.y / len;
    out.z = in.z / len;
    return out;
}

SpinImage::cpu::float3 operator* (SpinImage::cpu::float3 vec, float other) {
    SpinImage::cpu::float3 out;
    out.x = vec.x * other;
    out.y = vec.y * other;
    out.z = vec.z * other;
    return out;
}

SpinImage::cpu::float3 operator*(float other, SpinImage::cpu::float3 vec) {
    return operator*(vec, other);
}

std::ostream & operator<<(std::ostream & os, SpinImage::cpu::float3 vec) {
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return os;
}