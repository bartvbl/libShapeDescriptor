#include <spinImage/libraryBuildSettings.h>
#include <sstream>
#include <spinImage/utilities/compression/CompressedFileWriter.h>
#include "rawDescriptorDumper.h"

void SpinImage::dump::raw::descriptors(
        const std::experimental::filesystem::path &outputDumpFile,
        const SpinImage::cpu::QUICCIImages &images) {
    const unsigned int imageWidthPixels = spinImageWidthPixels;

    size_t imageBlockSize = images.imageCount * sizeof(QuiccImage);
    size_t outFileBufferSize = 5 + sizeof(size_t) + sizeof(unsigned int) + 2 * imageBlockSize;
    char* outFileBuffer = new char[outFileBufferSize];
    
    const std::string header = "QUIC";
    
    std::copy(header.begin(), header.end(), outFileBuffer);
    outFileBuffer[4] = 0;
    *reinterpret_cast<size_t*>(outFileBuffer + 5) = images.imageCount;
    *reinterpret_cast<unsigned int*>(outFileBuffer + 5 + sizeof(size_t)) = imageWidthPixels;
    std::copy(images.horizontallyIncreasingImages, images.horizontallyIncreasingImages + images.imageCount,
            reinterpret_cast<QuiccImage*>(outFileBuffer + 5 + sizeof(size_t) + sizeof(unsigned int)));
    std::copy(images.horizontallyDecreasingImages, images.horizontallyDecreasingImages + images.imageCount,
            reinterpret_cast<QuiccImage*>(outFileBuffer + 5 + sizeof(size_t) + sizeof(unsigned int) + imageBlockSize));

    SpinImage::utilities::writeCompressedFile(outFileBuffer, outFileBufferSize, outputDumpFile);
    
    delete[] outFileBuffer;
}