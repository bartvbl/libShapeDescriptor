#include <fstream>
#include <snappy.h>
#include <cassert>
#include "fileutils.h"

std::vector<std::experimental::filesystem::path> SpinImage::utilities::listDirectory(const std::string& directory) {
    std::vector<std::experimental::filesystem::path> foundFiles;

    for(auto &path : std::experimental::filesystem::directory_iterator(directory)) {
        foundFiles.emplace_back(path);
    }

    return foundFiles;
}

const char *SpinImage::utilities::readCompressedFile(const std::experimental::filesystem::path &archiveFile) {
    std::array<char, 5> headerTitle = {0, 0, 0, 0, 0};
    size_t compressedBufferSize;
    size_t decompressedBufferSize;

    std::ifstream decompressStream(archiveFile.string(), std::ios::out | std::ios::binary);

    decompressStream.read(headerTitle.data(), 4);
    decompressStream.read((char*) &decompressedBufferSize, sizeof(unsigned long));
    decompressStream.read((char*) &compressedBufferSize, sizeof(unsigned long));

    char* compressedBuffer = new char[compressedBufferSize];
    char* decompressedBuffer = new char[decompressedBufferSize];

    assert(std::string(headerTitle.data()) == "CDXF");

    decompressStream.read(compressedBuffer, compressedBufferSize);

    snappy::RawUncompress(compressedBuffer, compressedBufferSize, decompressedBuffer);

    delete[] compressedBuffer;

    return decompressedBuffer;
}

void SpinImage::utilities::writeCompressedFile(const char *buffer, size_t bufferSize, const std::experimental::filesystem::path &archiveFile) {
    std::experimental::filesystem::create_directories(archiveFile.parent_path());

    char* compressedBuffer = new char[snappy::MaxCompressedLength(bufferSize)];
    unsigned long compressedBufferSize = 0;
    snappy::RawCompress(buffer, bufferSize, compressedBuffer, &compressedBufferSize);

    const char header[5] = "CDXF";

    std::fstream outStream = std::fstream(archiveFile.string(), std::ios::out | std::ios::binary);

    outStream.write(header, 5 * sizeof(char));
    outStream.write((char*) &bufferSize, sizeof(size_t));
    outStream.write((char*) &compressedBufferSize, sizeof(size_t));
    outStream.write(compressedBuffer, compressedBufferSize);

    outStream.close();

    delete[] compressedBuffer;
}
