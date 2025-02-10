#pragma once

#include <ostream>
#include <string>
#include <cmath>
#include "float3.h"


namespace ShapeDescriptor {
    namespace cpu {
        struct double2 {
            double x = 0;
            double y = 0;

            double2() = default;
            double2(double x, double y) : x(x), y(y) {}

            double2 operator- (double2 other) {
                double2 out;
                out.x = x - other.x;
                out.y = y - other.y;
                return out;
            }

            double2 operator+ (double2 other) {
                double2 out;
                out.x = x + other.x;
                out.y = y + other.y;
                return out;
            }

            double2 operator* (double2 other) {
                double2 out;
                out.x = other.x * x;
                out.y = other.y * y;
                return out;
            }

            double2 operator/ (double divisor) {
                double2 out;
                out.x = x / divisor;
                out.y = y / divisor;
                return out;
            }

            bool operator== (const double2 &other) const {
                return
                        (x == other.x) &&
                        (y == other.y);
            }

            void operator+= (double2 other) {
                x += other.x;
                y += other.y;
            }

            void operator-= (double2 other) {
                x -= other.x;
                y -= other.y;
            }

            std::string to_string() {
                return "(" + std::to_string(x) + ", " + std::to_string(y) + ")";
            }
        };
    }
}

// Allow inclusion into std::set
namespace std {
    template <> struct hash<ShapeDescriptor::cpu::double2>
    {
        size_t operator()(const ShapeDescriptor::cpu::double2& p) const
        {
            return std::hash<double>()(p.x) ^ std::hash<double>()(p.y);
        }
    };
}


inline ShapeDescriptor::cpu::double2 make_double2_cpu(double x, double y, double z) {
    ShapeDescriptor::cpu::double2 out;
    out.x = x;
    out.y = y;
    return out;
}

inline double length(ShapeDescriptor::cpu::double2 vec) {
    return std::sqrt(vec.x * vec.x + vec.y * vec.y);
}

inline float dot(ShapeDescriptor::cpu::double2 a, ShapeDescriptor::cpu::double2 b) {
    return a.x * b.x + a.y * b.y;
}

inline ShapeDescriptor::cpu::double2 normalize(ShapeDescriptor::cpu::double2 in) {
    ShapeDescriptor::cpu::double2 out;
    double len = length(in);
    out.x = in.x / len;
    out.y = in.y / len;
    return out;
}

inline ShapeDescriptor::cpu::double2 operator* (ShapeDescriptor::cpu::double2 vec, double other) {
    ShapeDescriptor::cpu::double2 out;
    out.x = vec.x * other;
    out.y = vec.y * other;
    return out;
}

inline ShapeDescriptor::cpu::double2 operator*(double other, ShapeDescriptor::cpu::double2 vec) {
    return operator*(vec, other);
}

inline std::ostream & operator<<(std::ostream & os, ShapeDescriptor::cpu::double2 vec) {
    os << "(" << vec.x << ", " << vec.y << ")";
    return os;
}
