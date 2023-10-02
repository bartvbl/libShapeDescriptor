#include <array>
#include "MeshHasher.h"

/*
 * License from code adapted from meshoptimizer (https://github.com/zeux/meshoptimizer)
 *
 * MIT License

    Copyright (c) 2016-2023 Arseny Kapoulkine

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 *
 */

namespace ShapeDescriptor {
    uint32_t hashRange(const char* key, size_t len)
    {
        // MurmurHash2
        const uint32_t m = 0x5bd1e995;
        const int32_t r = 24;

        uint32_t h = 0;

        while (len >= 4)
        {
            uint32_t k = *reinterpret_cast<const uint32_t*>(key);

            k *= m;
            k ^= k >> r;
            k *= m;

            h *= m;
            h ^= k;

            key += 4;
            len -= 4;
        }

        return h;
    }

    bool rotateTriangle(std::array<cpu::float3, 3>& t)
    {
        int c01 = memcmp(&t[0], &t[1], sizeof(cpu::float3));
        int c02 = memcmp(&t[0], &t[2], sizeof(cpu::float3));
        int c12 = memcmp(&t[1], &t[2], sizeof(cpu::float3));

        if (c12 < 0 && c01 > 0)
        {
            // 1 is minimum, rotate 012 => 120
            cpu::float3 tv = t[0];
            t[0] = t[1];
            t[1] = t[2];
            t[2] = tv;
        }
        else if (c02 > 0 && c12 > 0)
        {
            // 2 is minimum, rotate 012 => 201
            cpu::float3 tv = t[2];
            t[2] = t[1];
            t[1] = t[0];
            t[0] = tv;
        }

        return c01 != 0 && c02 != 0 && c12 != 0;
    }

    uint32_t hashMesh(const ShapeDescriptor::cpu::Mesh &mesh)
    {
        size_t triangle_count = mesh.vertexCount / 3;

        const cpu::float3* vertices = mesh.vertices;

        uint32_t h1 = 0;
        uint32_t h2 = 0;

        for (size_t i = 0; i < triangle_count; ++i)
        {
            std::array<cpu::float3, 3> triangle;
            triangle.at(0) = vertices[i * 3 + 0];
            triangle.at(1) = vertices[i * 3 + 1];
            triangle.at(2) = vertices[i * 3 + 2];

            // skip degenerate triangles since some algorithms don't preserve them
            if (rotateTriangle(triangle))
            {
                uint32_t hash = hashRange(reinterpret_cast<char*>(triangle.data()), sizeof(triangle));

                h1 ^= hash;
                h2 += hash;
            }
        }

        return h1 * 0x5bd1e995 + h2;
    }
}
