#pragma once

#include "float3_cpu.h"

typedef struct float2_cpu {
    float x;
    float y;

    float2_cpu operator- (float2_cpu other) {
        float2_cpu out;
        out.x = x - other.x;
        out.y = y - other.y;
        return out;
    }

    float2_cpu operator* (float other) {
        float2_cpu out;
        out.x = other * x;
        out.y = other * y;
        return out;
    }

    float2_cpu operator/ (float other) {
        float2_cpu out;
        out.x = x / other;
        out.y = y / other;
        return out;
    }
} float2_cpu;

float length(float2_cpu vec);

inline float2_cpu make_float2_cpu(float x, float y) {
    float2_cpu out;
    out.x = x;
    out.y = y;
    return out;
}

float2_cpu normalize(float2_cpu in);