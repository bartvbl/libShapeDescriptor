#include <cmath>
#include "float3_cpu.h"
#include "float2_cpu.h"


float2_cpu to_float2(float3_cpu vec) {
    return make_float2_cpu(vec.x, vec.y);
}


float length(float3_cpu vec) {
    return std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

float3_cpu normalize(float3_cpu in) {
    float3_cpu out;
    float len = length(in);
    out.x = in.x / len;
    out.y = in.y / len;
    out.z = in.z / len;
    return out;
}

float3_cpu operator* (float3_cpu vec, float other) {
    float3_cpu out;
    out.x = vec.x * other;
    out.y = vec.y * other;
    out.z = vec.z * other;
    return out;
}

float3_cpu operator*(float other, float3_cpu vec) {
    return operator*(vec, other);
}

std::ostream & operator<<(std::ostream & os, float3_cpu vec) {
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return os;
}