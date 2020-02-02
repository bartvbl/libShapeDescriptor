#pragma once

#include <spinImage/cpu/types/Mesh.h>

namespace SpinImage {
    namespace dump {
        void mesh(cpu::Mesh mesh, std::string outputFile);
        void mesh(cpu::Mesh mesh, std::string outputFilePath,
                size_t highlightStartVertex, size_t highlightEndVertex);
    }
}