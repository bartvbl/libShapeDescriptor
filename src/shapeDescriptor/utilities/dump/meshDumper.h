#pragma once

#include <shapeDescriptor/cpu/types/Mesh.h>
#include <experimental/filesystem>
#include <vector_types.h>
#include <shapeDescriptor/cpu/types/array.h>

namespace ShapeDescriptor {
    namespace dump {
        void mesh(cpu::Mesh mesh, const std::experimental::filesystem::path outputFile);
        void mesh(cpu::Mesh mesh, const std::experimental::filesystem::path outputFilePath,
                size_t highlightStartVertex, size_t highlightEndVertex);
        void mesh(cpu::Mesh mesh, const std::experimental::filesystem::path &outputFilePath,
                ShapeDescriptor::cpu::array<float2> vertexTextureCoordinates, std::string textureMapPath);
    }
}