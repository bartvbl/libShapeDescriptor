#pragma once

#include "float3.h"

namespace SpinImage {
    namespace cpu {

        struct float2 {
            float x;
            float y;
        
            float2 operator- (float2 other) {
                float2 out;
                out.x = x - other.x;
                out.y = y - other.y;
                return out;
            }
        
            float2 operator* (float other) {
                float2 out;
                out.x = other * x;
                out.y = other * y;
                return out;
            }
        
            float2 operator/ (float other) {
                float2 out;
                out.x = x / other;
                out.y = y / other;
                return out;
            }
        };
    }
}

float length(SpinImage::cpu::float2 vec);

inline SpinImage::cpu::float2 make_float2_cpu(float x, float y) {
    SpinImage::cpu::float2 out;
    out.x = x;
    out.y = y;
    return out;
}

SpinImage::cpu::float2 normalize(SpinImage::cpu::float2 in);