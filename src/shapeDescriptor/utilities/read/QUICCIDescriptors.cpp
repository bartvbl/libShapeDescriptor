#include <shapeDescriptor/libraryBuildSettings.h>
#include "QUICCIDescriptors.h"
#include <exception>
#include <iostream>
#include <shapeDescriptor/utilities/fileutils.h>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>
#include <shapeDescriptor/cpu/types/array.h>

ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> readImageLZFile(const std::experimental::filesystem::path &path, unsigned int decompressionThreadCount) {
    size_t bufferSize;
    const char* inputBuffer = ShapeDescriptor::utilities::readCompressedFile(path, &bufferSize, decompressionThreadCount);

    char header[5] = {inputBuffer[0], inputBuffer[1], inputBuffer[2], inputBuffer[3], '\0'};
    if(std::string(header) != "QUIC") {
        std::cout << "WARNING: File header does not match expectations, and is thus possibly corrupt." << std::endl;
    }

    size_t imageCount = *reinterpret_cast<const size_t*>(inputBuffer + 5);
    unsigned int descriptorWidthPixels = *reinterpret_cast<const unsigned int*>(inputBuffer + 5 + sizeof(size_t));

    //std::cout << "\tFile has " << imageCount << " images" << std::endl;
    if(descriptorWidthPixels != spinImageWidthPixels) {
        std::cout << "The libSpinImage library was compiled with a different image size compared to those stored in this file." << std::endl;
        std::cout << "This means any processing this program does on them will not work correctly." << std::endl;
        std::cout << "Aborting index construction.." << std::endl;
        throw std::runtime_error("Invalid input file detected!");
    }

    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> images;
    images.length = imageCount;
    images.content = new ShapeDescriptor::QUICCIDescriptor[imageCount];

    const ShapeDescriptor::QUICCIDescriptor* imagesBasePointer
        = reinterpret_cast<const ShapeDescriptor::QUICCIDescriptor*>(inputBuffer + 5 + sizeof(size_t) + sizeof(unsigned int));

    std::copy(imagesBasePointer, imagesBasePointer + imageCount, images.content);

    delete[] inputBuffer;
    return images;
}

ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> ShapeDescriptor::read::QUICCIDescriptors(const std::experimental::filesystem::path &dumpFileLocation, unsigned int decompressionThreadCount) {
    return readImageLZFile(dumpFileLocation, decompressionThreadCount);
}


