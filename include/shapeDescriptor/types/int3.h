#pragma once

namespace ShapeDescriptor {
    namespace cpu {
        struct int3 {
            int32_t x = 0;
            int32_t y = 0;
            int32_t z = 0;

            int3() = default;
            int3(int32_t x, int32_t y, int32_t z) : x(x), y(y), z(z) {}

            [[nodiscard]] std::string to_string() const {
                return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
            }
        };
    }
}