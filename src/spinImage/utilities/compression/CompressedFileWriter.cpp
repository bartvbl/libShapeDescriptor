#include "CompressedFileWriter.h"
#include <fast-lzma2.h>
#include <spinImage/libraryBuildSettings.h>
#include <fstream>

void SpinImage::utilities::writeCompressedFile(const char *buffer, size_t bufferSize, const std::experimental::filesystem::path &archiveFile) {

    std::experimental::filesystem::create_directories(archiveFile.parent_path());

    const size_t maxCompressedBufferSize = FL2_compressBound(bufferSize);
    char* compressedBuffer = new char[maxCompressedBufferSize];
    unsigned long compressedBufferSize;
    //   #pragma omp critical
    {
        compressedBufferSize =
                FL2_compress(
                        (void*) compressedBuffer, maxCompressedBufferSize,
                        (void*) buffer, bufferSize,
                        LZMA2_COMPRESSION_LEVEL);
    }

    const char header[5] = "CDXF";

    std::fstream outStream = std::fstream(archiveFile.string(), std::ios::out | std::ios::binary);

    outStream.write(header, 5 * sizeof(char));
    outStream.write((char*) &bufferSize, sizeof(size_t));
    outStream.write((char*) &compressedBufferSize, sizeof(size_t));
    outStream.write(compressedBuffer, compressedBufferSize);

    outStream.close();

    delete[] compressedBuffer;
}
