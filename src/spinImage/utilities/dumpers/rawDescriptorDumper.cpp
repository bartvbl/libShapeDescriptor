#include <spinImage/libraryBuildSettings.h>
#include <sstream>
#include "rawDescriptorDumper.h"

void SpinImage::dump::raw::descriptors(
        const std::string &outputDumpFile,
        const SpinImage::cpu::QUICCIImages &images) {
    const size_t bytesPerQUICCImage = ((spinImageWidthPixels * spinImageWidthPixels) / 32) * sizeof(unsigned int);
    const unsigned int imageWidthPixels = spinImageWidthPixels;

    size_t imageBlockSize = images.imageCount * bytesPerQUICCImage;
    size_t outFileBufferSize = 5 + sizeof(size_t) + sizeof(unsigned int) + 2 * imageBlockSize;
    char* outFileBuffer = new char[outFileBufferSize];
    
    const std::string header = "QUIC";
    
    std::copy(header.begin(), header.end(), outFileBuffer);
    outFileBuffer[4] = 0;
    *reinterpret_cast<size_t*>(outFileBuffer + 5) = images.imageCount;
    *reinterpret_cast<unsigned int*>(outFileBuffer + 5 + sizeof(size_t)) = imageWidthPixels;
    std::copy(images.horizontallyIncreasingImages, images.horizontallyIncreasingImages + imageBlockSize, 
            outFileBuffer + 5 + sizeof(size_t) + sizeof(unsigned int));
    std::copy(images.horizontallyDecreasingImages, images.horizontallyDecreasingImages + imageBlockSize,
              outFileBuffer + 5 + sizeof(size_t) + sizeof(unsigned int) + imageBlockSize);

    SpinImage::utilities::writeCompressedFile(outFileBuffer, outFileBufferSize, outputDumpFile);
    
    delete[] outFileBuffer;
}