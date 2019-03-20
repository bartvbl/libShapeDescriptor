#pragma once

#include <ostream>

typedef struct float3_cpu {
    float x;
    float y;
    float z;

    float3_cpu operator- (float3_cpu other) {
        float3_cpu out;
        out.x = x - other.x;
        out.y = y - other.y;
        out.z = z - other.z;
        return out;
    }

    float3_cpu operator+ (float3_cpu other) {
        float3_cpu out;
        out.x = x + other.x;
        out.y = y + other.y;
        out.z = z + other.z;
        return out;
    }

    float3_cpu operator* (float3_cpu other) {
        float3_cpu out;
        out.x = other.x * x;
        out.y = other.y * y;
        out.z = other.z * z;
        return out;
    }

    float3_cpu operator/ (float divisor) {
        float3_cpu out;
        out.x = x / divisor;
        out.y = y / divisor;
        out.z = z / divisor;
        return out;
    }

    bool operator== (float3_cpu other) {
        return
                (x == other.x) &&
                (y == other.y) &&
                (z == other.z);
    }
} float3_cpu;


inline float3_cpu make_float3_cpu(float x, float y, float z) {
    float3_cpu out;
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
}

float3_cpu normalize(float3_cpu in);
std::ostream & operator<<(std::ostream & os, float3_cpu vec);
float3_cpu operator* (float3_cpu vec, float other);
float3_cpu operator* (float other, float3_cpu vec);
float length(float3_cpu vec);
