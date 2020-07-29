#include <spinImage/libraryBuildSettings.h>
#include <sstream>
#include "rawDescriptorDumper.h"
#include <spinImage/utilities/fileutils.h>
#include <spinImage/gpu/types/methods/QUICCIDescriptor.h>
#include <spinImage/cpu/types/array.h>

void SpinImage::dump::raw::descriptors(
        const std::experimental::filesystem::path &outputDumpFile,
        const SpinImage::cpu::array<SpinImage::gpu::QUICCIDescriptor> &images) {
    const unsigned int imageWidthPixels = spinImageWidthPixels;

    size_t imageBlockSize = images.length * sizeof(QuiccImage);
    size_t outFileBufferSize = 5 + sizeof(size_t) + sizeof(unsigned int) + 2 * imageBlockSize;
    char* outFileBuffer = new char[outFileBufferSize];
    
    const std::string header = "QUIC";
    
    std::copy(header.begin(), header.end(), outFileBuffer);
    outFileBuffer[4] = 0;
    *reinterpret_cast<size_t*>(outFileBuffer + 5) = images.length;
    *reinterpret_cast<unsigned int*>(outFileBuffer + 5 + sizeof(size_t)) = imageWidthPixels;
    std::copy(images.content, images.content + images.length,
            reinterpret_cast<SpinImage::gpu::QUICCIDescriptor*>(outFileBuffer + 5 + sizeof(size_t) + sizeof(unsigned int)));

    SpinImage::utilities::writeCompressedFile(outFileBuffer, outFileBufferSize, outputDumpFile);
    
    delete[] outFileBuffer;
}