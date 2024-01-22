#pragma once

#include <filesystem>
#include <shapeDescriptor/containerTypes.h>

namespace ShapeDescriptor {
    template<typename DescriptorType>
    void writeCompressedDescriptors(
            const std::filesystem::path &outputDumpFile,
            const ShapeDescriptor::cpu::array<DescriptorType> &images,
            unsigned int compressionThreadCount) {
        const unsigned int imageWidthPixels = spinImageWidthPixels;

        size_t imageBlockSize = images.length * sizeof(DescriptorType);
        size_t outFileBufferSize = 5 + sizeof(size_t) + sizeof(unsigned int) + 2 * imageBlockSize;
        char *outFileBuffer = new char[outFileBufferSize];

        const std::string header = "CSDF";

        std::copy(header.begin(), header.end(), outFileBuffer);
        outFileBuffer[4] = 0;
        *reinterpret_cast<size_t *>(outFileBuffer + 5) = images.length;
        *reinterpret_cast<unsigned int *>(outFileBuffer + 5 + sizeof(size_t)) = imageWidthPixels;
        std::copy(images.content, images.content + images.length,
                  reinterpret_cast<DescriptorType*>(outFileBuffer + 5 + sizeof(size_t) +
                                                                        sizeof(unsigned int)));

        ShapeDescriptor::writeCompressedFile(outFileBuffer, outFileBufferSize, outputDumpFile, compressionThreadCount);

        delete[] outFileBuffer;
    }



}