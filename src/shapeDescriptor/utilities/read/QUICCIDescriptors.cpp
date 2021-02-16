#include <shapeDescriptor/libraryBuildSettings.h>
#include "QUICCIDescriptors.h"
#include <exception>
#include <iostream>
#include <shapeDescriptor/utilities/fileutils.h>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>
#include <shapeDescriptor/cpu/types/array.h>

ShapeDescriptor::QUICCIDescriptorFileHeader readHeader(const char* startOfFileBuffer) {
    ShapeDescriptor::QUICCIDescriptorFileHeader header;
    header.fileID = {startOfFileBuffer[0], startOfFileBuffer[1], startOfFileBuffer[2], startOfFileBuffer[3], '\0'};
    if(std::string(header.fileID.data()) != "QUIC") {
        std::cout << "WARNING: File header does not match expectations, and is thus possibly corrupt." << std::endl;
    }

    size_t imageCount = *reinterpret_cast<const size_t*>(startOfFileBuffer + 5);
    unsigned int descriptorWidthPixels = *reinterpret_cast<const unsigned int*>(startOfFileBuffer + 5 + sizeof(size_t));

    header.imageCount = imageCount;
    header.descriptorWidthPixels = descriptorWidthPixels;

    return header;
}

ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> readImageLZFile(const std::experimental::filesystem::path &path, unsigned int decompressionThreadCount) {
    size_t bufferSize;
    const char* inputBuffer = ShapeDescriptor::utilities::readCompressedFile(path, &bufferSize, decompressionThreadCount);

    ShapeDescriptor::QUICCIDescriptorFileHeader header = readHeader(inputBuffer);

    //std::cout << "\tFile has " << imageCount << " images" << std::endl;
    if(header.descriptorWidthPixels != spinImageWidthPixels) {
        std::cout << "The libSpinImage library was compiled with a different image size compared to those stored in this file." << std::endl;
        std::cout << "This means any processing this program does on them will not work correctly." << std::endl;
        std::cout << "Aborting index construction.." << std::endl;
        throw std::runtime_error("Invalid input file detected!");
    }

    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> images;
    images.length = header.imageCount;
    images.content = new ShapeDescriptor::QUICCIDescriptor[header.imageCount];

    const ShapeDescriptor::QUICCIDescriptor* imagesBasePointer
        = reinterpret_cast<const ShapeDescriptor::QUICCIDescriptor*>(inputBuffer + sizeof(ShapeDescriptor::QUICCIDescriptorFileHeader));

    std::copy(imagesBasePointer, imagesBasePointer + header.imageCount, images.content);

    delete[] inputBuffer;
    return images;
}

ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> ShapeDescriptor::read::QUICCIDescriptors(const std::experimental::filesystem::path &dumpFileLocation, unsigned int decompressionThreadCount) {
    return readImageLZFile(dumpFileLocation, decompressionThreadCount);
}

ShapeDescriptor::QUICCIDescriptorFileHeader
ShapeDescriptor::read::QuicciDescriptorFileHeader(const std::experimental::filesystem::path &path) {
    size_t bufferSize;
    const size_t headerSize = sizeof(ShapeDescriptor::QUICCIDescriptorFileHeader);
    const char* inputBuffer = ShapeDescriptor::utilities::readCompressedFileUpToNBytes(path, &bufferSize, headerSize, 1);

    ShapeDescriptor::QUICCIDescriptorFileHeader header = readHeader(inputBuffer);

    delete[] inputBuffer;

    return header;
}


