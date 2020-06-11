#pragma once

#include <ostream>


namespace SpinImage {
    namespace cpu {
        struct float3 {
            float x;
            float y;
            float z;

            float3 operator- (float3 other) {
                float3 out;
                out.x = x - other.x;
                out.y = y - other.y;
                out.z = z - other.z;
                return out;
            }

            float3 operator+ (float3 other) {
                float3 out;
                out.x = x + other.x;
                out.y = y + other.y;
                out.z = z + other.z;
                return out;
            }

            float3 operator* (float3 other) {
                float3 out;
                out.x = other.x * x;
                out.y = other.y * y;
                out.z = other.z * z;
                return out;
            }

            float3 operator/ (float divisor) {
                float3 out;
                out.x = x / divisor;
                out.y = y / divisor;
                out.z = z / divisor;
                return out;
            }

            bool operator== (float3 other) {
                return
                        (x == other.x) &&
                        (y == other.y) &&
                        (z == other.z);
            }

            std::string to_string() {
                return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
            }
        };
    }
}


inline SpinImage::cpu::float3 make_float3_cpu(float x, float y, float z) {
    SpinImage::cpu::float3 out;
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
}

SpinImage::cpu::float3 normalize(SpinImage::cpu::float3 in);
std::ostream & operator<<(std::ostream & os, SpinImage::cpu::float3 vec);
SpinImage::cpu::float3 operator* (SpinImage::cpu::float3 vec, float other);
SpinImage::cpu::float3 operator* (float other, SpinImage::cpu::float3 vec);
float length(SpinImage::cpu::float3 vec);
