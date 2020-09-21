#include <shapeDescriptor/libraryBuildSettings.h>
#include <sstream>
#include <shapeDescriptor/utilities/fileutils.h>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>
#include <shapeDescriptor/cpu/types/array.h>
#include "QUICCIDescriptors.h"

void ShapeDescriptor::dump::raw::QUICCIDescriptors(
        const std::experimental::filesystem::path &outputDumpFile,
        const ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> &images) {
    const unsigned int imageWidthPixels = spinImageWidthPixels;

    size_t imageBlockSize = images.length * sizeof(ShapeDescriptor::QUICCIDescriptor);
    size_t outFileBufferSize = 5 + sizeof(size_t) + sizeof(unsigned int) + 2 * imageBlockSize;
    char* outFileBuffer = new char[outFileBufferSize];
    
    const std::string header = "QUIC";
    
    std::copy(header.begin(), header.end(), outFileBuffer);
    outFileBuffer[4] = 0;
    *reinterpret_cast<size_t*>(outFileBuffer + 5) = images.length;
    *reinterpret_cast<unsigned int*>(outFileBuffer + 5 + sizeof(size_t)) = imageWidthPixels;
    std::copy(images.content, images.content + images.length,
            reinterpret_cast<ShapeDescriptor::QUICCIDescriptor*>(outFileBuffer + 5 + sizeof(size_t) + sizeof(unsigned int)));

    ShapeDescriptor::utilities::writeCompressedFile(outFileBuffer, outFileBufferSize, outputDumpFile);
    
    delete[] outFileBuffer;
}