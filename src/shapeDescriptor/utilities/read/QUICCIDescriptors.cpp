#include <exception>
#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>

ShapeDescriptor::QUICCIDescriptorFileHeader readHeader(std::vector<char>& startOfFileBuffer) {
    ShapeDescriptor::QUICCIDescriptorFileHeader header;
    header.fileID = {startOfFileBuffer[0], startOfFileBuffer[1], startOfFileBuffer[2], startOfFileBuffer[3]};
    if(std::string(header.fileID.data()) != "QUIC") {
        std::cout << "WARNING: File header does not match expectations, and is thus possibly corrupt." << std::endl;
    }

    size_t imageCount = *reinterpret_cast<const size_t*>(startOfFileBuffer.data() + 5);
    unsigned int descriptorWidthPixels = *reinterpret_cast<const unsigned int*>(startOfFileBuffer.data() + 5 + sizeof(size_t));

    header.imageCount = imageCount;
    header.descriptorWidthPixels = descriptorWidthPixels;

    return header;
}

ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> readImageLZFile(const std::filesystem::path &path, unsigned int decompressionThreadCount) {
    std::vector<char> inputBuffer = ShapeDescriptor::readCompressedFile(path, decompressionThreadCount);

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

    const size_t headerSize = 5 * sizeof(char) + sizeof(size_t) + sizeof(unsigned int);
    const ShapeDescriptor::QUICCIDescriptor* imagesBasePointer
        = reinterpret_cast<const ShapeDescriptor::QUICCIDescriptor*>(inputBuffer.data() + headerSize);

    std::copy(imagesBasePointer, imagesBasePointer + header.imageCount, images.content);

    return images;
}

ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> ShapeDescriptor::readCompressedQUICCIDescriptors(
        const std::filesystem::path &dumpFileLocation, unsigned int decompressionThreadCount) {
    return readImageLZFile(dumpFileLocation, decompressionThreadCount);
}

ShapeDescriptor::QUICCIDescriptorFileHeader
ShapeDescriptor::readCompressedQUICCIDescriptorFileHeader(const std::filesystem::path &dumpFileLocation) {
    const size_t headerSize = sizeof(ShapeDescriptor::QUICCIDescriptorFileHeader);
    std::vector<char> inputBuffer = ShapeDescriptor::readCompressedFileUpToNBytes(dumpFileLocation, headerSize, 1);

    ShapeDescriptor::QUICCIDescriptorFileHeader header = readHeader(inputBuffer);

    return header;
}


