#include <spinImage/libraryBuildSettings.h>
#include "quicciReader.h"
#include <exception>
#include <ZipLib/ZipArchive.h>
#include <ZipLib/ZipFile.h>
#include <spinImage/utilities/fileutils.h>

SpinImage::cpu::QUICCIImages readImageLZFile(const std::experimental::filesystem::path &path) {
    size_t bufferSize;
    const char* inputBuffer = SpinImage::utilities::readCompressedFile(path, &bufferSize);

    char header[5] = {inputBuffer[0], inputBuffer[1], inputBuffer[2], inputBuffer[3], '\0'};
    if(std::string(header) != "QUIC") {
        std::cout << "WARNING: File header does not match expectations, and is thus possibly corrupt." << std::endl;
    }

    size_t imageCount = *reinterpret_cast<const size_t*>(inputBuffer + 4);
    unsigned int descriptorWidthPixels = *reinterpret_cast<const unsigned int*>(inputBuffer + 4 + sizeof(size_t));

    //std::cout << "\tFile has " << imageCount << " images" << std::endl;
    if(descriptorWidthPixels != spinImageWidthPixels) {
        std::cout << "The libSpinImage library was compiled with a different image size compared to those stored in this file." << std::endl;
        std::cout << "This means any processing this program does on them will not work correctly." << std::endl;
        std::cout << "Aborting index construction.." << std::endl;
        throw std::runtime_error("Invalid input file detected!");
    }

    const unsigned int uintsPerQUICCImage = (spinImageWidthPixels * spinImageWidthPixels) / 32;
    SpinImage::cpu::QUICCIImages images;
    images.imageCount = imageCount;
    const size_t imageBufferLength = uintsPerQUICCImage * imageCount;
    images.horizontallyIncreasingImages = new unsigned int[imageBufferLength];
    images.horizontallyDecreasingImages = new unsigned int[imageBufferLength];

    const unsigned int* horizontallyIncreasingBasePointer
        = reinterpret_cast<const unsigned int*>(inputBuffer + 4 + sizeof(size_t) + sizeof(unsigned int));
    const unsigned int* horizontallyDecreasingBasePointer
            = reinterpret_cast<const unsigned int*>(inputBuffer + 4 + sizeof(size_t) + sizeof(unsigned int)
                    + imageBufferLength * sizeof(unsigned int));

    std::copy(horizontallyIncreasingBasePointer, horizontallyIncreasingBasePointer + imageBufferLength,
            images.horizontallyIncreasingImages);
    std::copy(horizontallyDecreasingBasePointer, horizontallyDecreasingBasePointer + imageBufferLength,
              images.horizontallyDecreasingImages);

    delete[] inputBuffer;
    return images;
}

SpinImage::cpu::QUICCIImages SpinImage::read::QUICCImagesFromDumpFile(const std::experimental::filesystem::path &dumpFileLocation) {
    return readImageLZFile(dumpFileLocation);
}


