#pragma once

#include <spinImage/cpu/types/Mesh.h>
#include <experimental/filesystem>
#include <vector_types.h>

namespace SpinImage {
    namespace dump {
        void mesh(cpu::Mesh mesh, const std::experimental::filesystem::path outputFile);
        void mesh(cpu::Mesh mesh, const std::experimental::filesystem::path outputFilePath,
                size_t highlightStartVertex, size_t highlightEndVertex);
        void mesh(cpu::Mesh mesh, const std::experimental::filesystem::path &outputFilePath,
                SpinImage::array<float2> vertexTextureCoordinates, std::string textureMapPath);
    }
}