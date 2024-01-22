#pragma once

#include <filesystem>
#include <iostream>
#include <shapeDescriptor/containerTypes.h>

namespace ShapeDescriptor {
    template<typename DescriptorType>
    void writeCompressedDescriptors(
            const std::filesystem::path &outputDumpFile,
            const ShapeDescriptor::cpu::array<DescriptorType> images,
            unsigned int compressionThreadCount = 1) {

        size_t imageBlockSize = images.length * sizeof(DescriptorType);
        size_t outFileBufferSize = 5 + sizeof(size_t) + 2 * imageBlockSize;
        char *outFileBuffer = new char[outFileBufferSize];

        const std::string header = "CSDF";

        std::copy(header.begin(), header.end(), outFileBuffer);
        outFileBuffer[4] = 0;
        *reinterpret_cast<size_t *>(outFileBuffer + 5) = images.length;
        std::copy(images.content, images.content + images.length,
                  reinterpret_cast<DescriptorType*>(outFileBuffer + 5 + sizeof(size_t)));

        ShapeDescriptor::writeCompressedFile(outFileBuffer, outFileBufferSize, outputDumpFile, compressionThreadCount);
        delete[] outFileBuffer;
    }

    template<typename DescriptorType>
    ShapeDescriptor::cpu::array<DescriptorType> readCompressedDescriptors(
            const std::filesystem::path &compressedFileLocation, unsigned int decompressionThreadCount = 1) {
        std::vector<char> inputBuffer = ShapeDescriptor::readCompressedFile(compressedFileLocation, decompressionThreadCount);

        std::array<char, 5> fileID {0, 0, 0, 0, 0};
        fileID = {inputBuffer.at(0), inputBuffer.at(1), inputBuffer.at(2), inputBuffer.at(3)};
        if(std::string(fileID.data()) != "CSDF") {
            std::cout << "WARNING: File header does not match expectations, and is thus possibly corrupt." << std::endl;
        }

        size_t imageCount = *reinterpret_cast<const size_t*>(inputBuffer.data() + sizeof(fileID));

        ShapeDescriptor::cpu::array<DescriptorType> images(imageCount);

        const size_t headerSize = 5 * sizeof(char) + sizeof(size_t);
        const DescriptorType* imagesBasePointer = reinterpret_cast<const DescriptorType*>(inputBuffer.data() + headerSize);

        std::copy(imagesBasePointer, imagesBasePointer + imageCount, images.content);

        return images;
    }



}