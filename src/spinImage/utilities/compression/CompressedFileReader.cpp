#include <fstream>
#include <cassert>
#include <fast-lzma2.h>
#include "CompressedFileReader.h"

const char *SpinImage::utilities::readCompressedFile(const std::experimental::filesystem::path &archiveFile, size_t* fileSizeBytes, bool enableMultithreading) {
    std::array<char, 5> headerTitle = {0, 0, 0, 0, 0};
    size_t compressedBufferSize;
    size_t decompressedBufferSize;

    if(!std::experimental::filesystem::exists(archiveFile)) {
        throw std::runtime_error("The file " + std::experimental::filesystem::absolute(archiveFile).string() + " was not found.");
    }

    std::ifstream decompressStream(archiveFile.string(), std::ios::in | std::ios::binary);

    decompressStream.read(headerTitle.data(), 5);
    decompressStream.read((char*) &decompressedBufferSize, sizeof(size_t));
    decompressStream.read((char*) &compressedBufferSize, sizeof(size_t));

    *fileSizeBytes = decompressedBufferSize;

    char* compressedBuffer = new char[compressedBufferSize];
    char* decompressedBuffer = new char[decompressedBufferSize];

    assert(std::string(headerTitle.data()) == "CDXF");

    decompressStream.read(compressedBuffer, compressedBufferSize);

    decompressStream.close();

    FL2_decompress(
            (void*) decompressedBuffer, decompressedBufferSize,
            (void*) compressedBuffer, compressedBufferSize);

    delete[] compressedBuffer;

    return decompressedBuffer;
}