#pragma once

#include <spinImage/cpu/types/Mesh.h>
#include <experimental/filesystem>

namespace SpinImage {
    namespace dump {
        void mesh(cpu::Mesh mesh, std::experimental::filesystem::path outputFile);
        void mesh(cpu::Mesh mesh, std::experimental::filesystem::path outputFilePath,
                size_t highlightStartVertex, size_t highlightEndVertex);
    }
}