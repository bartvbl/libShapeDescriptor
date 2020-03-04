#include <spinImage/libraryBuildSettings.h>
#include "quicciReader.h"
#include <exception>
#include <ZipLib/ZipArchive.h>
#include <ZipLib/ZipFile.h>
#include <spinImage/utilities/fileutils.h>

SpinImage::cpu::QUICCIImages readImageLZFile(const std::experimental::filesystem::path &path) {
    size_t bufferSize;
    const char* inputBuffer = SpinImage::utilities::readCompressedFile(path, &bufferSize, false);

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

    SpinImage::cpu::QUICCIImages images;
    images.imageCount = imageCount;
    images.horizontallyIncreasingImages = new QuiccImage[imageCount];
    images.horizontallyDecreasingImages = new QuiccImage[imageCount];

    const QuiccImage* horizontallyIncreasingBasePointer
        = reinterpret_cast<const QuiccImage*>(inputBuffer + 4 + sizeof(size_t) + sizeof(unsigned int));
    const QuiccImage* horizontallyDecreasingBasePointer
        = reinterpret_cast<const QuiccImage*>(inputBuffer + 4 + sizeof(size_t) + sizeof(unsigned int) + imageCount * sizeof(QuiccImage));

    std::copy(horizontallyIncreasingBasePointer, horizontallyIncreasingBasePointer + imageCount,
            images.horizontallyIncreasingImages);
    std::copy(horizontallyDecreasingBasePointer, horizontallyDecreasingBasePointer + imageCount,
            images.horizontallyDecreasingImages);

    delete[] inputBuffer;
    return images;
}

SpinImage::cpu::QUICCIImages SpinImage::read::QUICCImagesFromDumpFile(const std::experimental::filesystem::path &dumpFileLocation) {
    return readImageLZFile(dumpFileLocation);
}


