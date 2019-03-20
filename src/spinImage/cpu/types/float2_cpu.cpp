#include <cmath>
#include "float2_cpu.h"

float length(float2_cpu vec) {
    return std::sqrt(vec.x * vec.x + vec.y * vec.y);
}

float2_cpu normalize(float2_cpu in) {
    float2_cpu out;
    float len = length(in);
    out.x = in.x / len;
    out.y = in.y / len;
    return out;
}