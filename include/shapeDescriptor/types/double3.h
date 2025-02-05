#pragma once

#include <ostream>
#include <string>
#include <cmath>
#include "float3.h"


namespace ShapeDescriptor {
    namespace cpu {
        struct double3 {
            double x = 0;
            double y = 0;
            double z = 0;

            double3() = default;
            double3(double x, double y, double z) : x(x), y(y), z(z) {}

            double3 operator- (double3 other) {
                double3 out;
                out.x = x - other.x;
                out.y = y - other.y;
                out.z = z - other.z;
                return out;
            }

            double3 operator+ (double3 other) {
                double3 out;
                out.x = x + other.x;
                out.y = y + other.y;
                out.z = z + other.z;
                return out;
            }

            double3 operator* (double3 other) {
                double3 out;
                out.x = other.x * x;
                out.y = other.y * y;
                out.z = other.z * z;
                return out;
            }

            double3 operator/ (double divisor) {
                double3 out;
                out.x = x / divisor;
                out.y = y / divisor;
                out.z = z / divisor;
                return out;
            }

            bool operator== (const double3 &other) const {
                return
                        (x == other.x) &&
                        (y == other.y) &&
                        (z == other.z);
            }

            void operator+= (double3 other) {
                x += other.x;
                y += other.y;
                z += other.z;
            }

            void operator-= (double3 other) {
                x -= other.x;
                y -= other.y;
                z -= other.z;
            }

            float3 as_float3() const {
                return make_float3_cpu(float(x), float(y), float(z));
            }

            std::string to_string() {
                return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
            }
        };
    }
}

// Allow inclusion into std::set
namespace std {
    template <> struct hash<ShapeDescriptor::cpu::double3>
    {
        size_t operator()(const ShapeDescriptor::cpu::double3& p) const
        {
            return std::hash<double>()(p.x) ^ std::hash<double>()(p.y) ^ std::hash<double>()(p.z);
        }
    };
}


inline ShapeDescriptor::cpu::double3 make_double3_cpu(double x, double y, double z) {
    ShapeDescriptor::cpu::double3 out;
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
}

inline double length(ShapeDescriptor::cpu::double3 vec) {
    return std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

inline ShapeDescriptor::cpu::double3 normalize(ShapeDescriptor::cpu::double3 in) {
    ShapeDescriptor::cpu::double3 out;
    double len = length(in);
    out.x = in.x / len;
    out.y = in.y / len;
    out.z = in.z / len;
    return out;
}

inline ShapeDescriptor::cpu::double3 cross(const ShapeDescriptor::cpu::double3 a, const ShapeDescriptor::cpu::double3 b) {
    double x = a.y * b.z - a.z * b.y;
    double y = a.z * b.x - a.x * b.z;
    double z = a.x * b.y - a.y * b.x;
    return {x, y, z};
}

inline ShapeDescriptor::cpu::double3 operator* (ShapeDescriptor::cpu::double3 vec, double other) {
    ShapeDescriptor::cpu::double3 out;
    out.x = vec.x * other;
    out.y = vec.y * other;
    out.z = vec.z * other;
    return out;
}

inline ShapeDescriptor::cpu::double3 operator*(double other, ShapeDescriptor::cpu::double3 vec) {
    return operator*(vec, other);
}

inline std::ostream & operator<<(std::ostream & os, ShapeDescriptor::cpu::double3 vec) {
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return os;
}
