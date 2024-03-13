#pragma once

#include <ostream>
#include <string>
#include <cmath>


namespace ShapeDescriptor {
    namespace cpu {
        struct float3 {
            float x = 0;
            float y = 0;
            float z = 0;

            float3() = default;
            float3(float x, float y, float z) : x(x), y(y), z(z) {}

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
            float3(::float3 const &in) : x(in.x), y(in.y), z(in.z) {}

            __host__ __device__ operator ::float3() const {
                return ::float3{x, y, z};
            }
#endif

            float3 operator- (float3 other) const {
                float3 out;
                out.x = x - other.x;
                out.y = y - other.y;
                out.z = z - other.z;
                return out;
            }

            float3 operator+ (float3 other) const {
                float3 out;
                out.x = x + other.x;
                out.y = y + other.y;
                out.z = z + other.z;
                return out;
            }

            float3 operator* (float3 other) const {
                float3 out;
                out.x = other.x * x;
                out.y = other.y * y;
                out.z = other.z * z;
                return out;
            }

            float3 operator/ (float divisor) const {
                float3 out;
                out.x = x / divisor;
                out.y = y / divisor;
                out.z = z / divisor;
                return out;
            }

            bool operator== (const float3 &other) const {
                return
                        (x == other.x) &&
                        (y == other.y) &&
                        (z == other.z);
            }

            bool operator< (const float3 &other) const {
                return (x < other.x) && (y < other.y) && (z < other.z);
            }

            void operator*= (float scaleFactor) {
                x *= scaleFactor;
                y *= scaleFactor;
                z *= scaleFactor;
            }

            void operator/= (float factor) {
                x /= factor;
                y /= factor;
                z /= factor;
            }

            void operator+= (float3 other) {
                x += other.x;
                y += other.y;
                z += other.z;
            }

            void operator-= (float3 other) {
                x -= other.x;
                y -= other.y;
                z -= other.z;
            }

            std::string to_string() {
                return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
            }
        };
    }
}

// Allow inclusion into std::set
namespace std {
    template <> struct hash<ShapeDescriptor::cpu::float3>
    {
        size_t operator()(const ShapeDescriptor::cpu::float3& p) const
        {
            return std::hash<float>()(p.x) ^ std::hash<float>()(p.y) ^ std::hash<float>()(p.z);
        }
    };
}


inline ShapeDescriptor::cpu::float3 make_float3_cpu(float x, float y, float z) {
    ShapeDescriptor::cpu::float3 out;
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
}

inline float length(ShapeDescriptor::cpu::float3 vec) {
    return std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

inline ShapeDescriptor::cpu::float3 normalize(ShapeDescriptor::cpu::float3 in) {
    ShapeDescriptor::cpu::float3 out;
    float len = length(in);
    out.x = in.x / len;
    out.y = in.y / len;
    out.z = in.z / len;
    return out;
}

inline ShapeDescriptor::cpu::float3 operator* (ShapeDescriptor::cpu::float3 vec, float other) {
    ShapeDescriptor::cpu::float3 out;
    out.x = vec.x * other;
    out.y = vec.y * other;
    out.z = vec.z * other;
    return out;
}

inline ShapeDescriptor::cpu::float3 operator*(float other, ShapeDescriptor::cpu::float3 vec) {
    return operator*(vec, other);
}

inline std::ostream & operator<<(std::ostream &os, const ShapeDescriptor::cpu::float3 vec) {
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return os;
}

inline float dot(ShapeDescriptor::cpu::float3 a, ShapeDescriptor::cpu::float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline ShapeDescriptor::cpu::float3 cross(const ShapeDescriptor::cpu::float3 a, const ShapeDescriptor::cpu::float3 b) {
    float x = a.y * b.z - a.z * b.y;
    float y = a.z * b.x - a.x * b.z;
    float z = a.x * b.y - a.y * b.x;
    return {x, y, z};
}