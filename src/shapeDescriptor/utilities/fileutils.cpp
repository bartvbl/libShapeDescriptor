#include <fstream>
#include <cassert>
#include "fileutils.h"
#include <algorithm>
#include <array>
#include <shapeDescriptor/utilities/compress/byteCompressor.h>



std::vector<std::experimental::filesystem::path> ShapeDescriptor::utilities::listDirectory(const std::string& directory) {
    std::vector<std::experimental::filesystem::path> foundFiles;

    for(auto &path : std::experimental::filesystem::directory_iterator(directory)) {
        foundFiles.emplace_back(path);
    }

    std::sort(foundFiles.begin(), foundFiles.end());

    return foundFiles;
}

const char *ShapeDescriptor::utilities::readCompressedFile(const std::experimental::filesystem::path &archiveFile, size_t* fileSizeBytes, unsigned int threadCount) {
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

//#pragma omp critical
    {
        //if(enableMultithreading) {

            //FL2_decompressDCtx(
            //        decompressionContext,
            //        (void*) decompressedBuffer, decompressedBufferSize,
            //        (void*) compressedBuffer, compressedBufferSize);
        //} else {
            ShapeDescriptor::utilities::decompressBytesMultithreaded(
                    (void*) decompressedBuffer, decompressedBufferSize,
                    (void*) compressedBuffer, compressedBufferSize,
                    threadCount);
        //}
    }

    delete[] compressedBuffer;

    return decompressedBuffer;
}

void ShapeDescriptor::utilities::writeCompressedFile(const char *buffer, size_t bufferSize, const std::experimental::filesystem::path &archiveFile, unsigned int threadCount) {

    std::experimental::filesystem::create_directories(std::experimental::filesystem::absolute(archiveFile).parent_path());

    const size_t maxCompressedBufferSize = ShapeDescriptor::utilities::computeMaxCompressedBufferSize(bufferSize);
    char* compressedBuffer = new char[maxCompressedBufferSize];
    unsigned long compressedBufferSize;
 //   #pragma omp critical
    {
        compressedBufferSize =
                ShapeDescriptor::utilities::compressBytesMultithreaded(
                        (void*) compressedBuffer, maxCompressedBufferSize,
                        (void*) buffer, bufferSize, threadCount);
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
