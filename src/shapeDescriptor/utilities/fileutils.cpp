#include <fstream>
#include <cassert>
#include "fileutils.h"
#include <algorithm>
#include <array>
#include <random>
#include <shapeDescriptor/utilities/compress/byteCompressor.h>



std::vector<std::filesystem::path> ShapeDescriptor::utilities::listDirectory(const std::filesystem::path& directory) {
    std::vector<std::filesystem::path> foundFiles;

    for(auto &path : std::filesystem::directory_iterator(directory)) {
        foundFiles.emplace_back(path);
    }

    std::sort(foundFiles.begin(), foundFiles.end());

    return foundFiles;
}

std::vector<std::filesystem::path> ShapeDescriptor::utilities::listDirectoryAndSubdirectories(const std::filesystem::path &directory) {
    std::vector<std::filesystem::path> foundFiles;

    for(auto &path : std::filesystem::recursive_directory_iterator(directory)) {
        if(path.exists() && path.is_regular_file()) {
            foundFiles.emplace_back(path);
        }
    }

    std::sort(foundFiles.begin(), foundFiles.end());

    return foundFiles;
}

const char* readLZMAFile(const std::filesystem::path &archiveFile, size_t* fileSizeBytes, size_t readLimit, unsigned int threadCount) {
    std::array<char, 5> headerTitle = {0, 0, 0, 0, 0};
    size_t compressedBufferSize;
    size_t decompressedBufferSize;

    if(!std::filesystem::exists(archiveFile)) {
        throw std::runtime_error("The file " + std::filesystem::absolute(archiveFile).string() + " was not found.");
    }

    std::ifstream decompressStream(archiveFile.string(), std::ios::in | std::ios::binary);

    decompressStream.read(headerTitle.data(), 5);
    decompressStream.read((char*) &decompressedBufferSize, sizeof(size_t));
    decompressStream.read((char*) &compressedBufferSize, sizeof(size_t));

    size_t numberOfDecompressedBytesToRead = std::min<size_t>(decompressedBufferSize, readLimit);

    *fileSizeBytes = numberOfDecompressedBytesToRead;

    char* compressedBuffer = new char[compressedBufferSize];
    char* decompressedBuffer = new char[numberOfDecompressedBytesToRead];

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
                    (void*) decompressedBuffer, numberOfDecompressedBytesToRead,
                    (void*) compressedBuffer, compressedBufferSize,
                    threadCount);
        //}
    }

    delete[] compressedBuffer;

    return decompressedBuffer;
}

const char *ShapeDescriptor::utilities::readCompressedFile(const std::filesystem::path &archiveFile, size_t* fileSizeBytes, unsigned int threadCount) {
    return readLZMAFile(archiveFile, fileSizeBytes, std::numeric_limits<size_t>::max(), threadCount);
}

void ShapeDescriptor::utilities::writeCompressedFile(const char *buffer, size_t bufferSize, const std::filesystem::path &archiveFile, unsigned int threadCount) {

    std::filesystem::create_directories(std::filesystem::absolute(archiveFile).parent_path());

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

const char *
ShapeDescriptor::utilities::readCompressedFileUpToNBytes(const std::filesystem::path &archiveFile,
                                                         size_t* readByteCount,
                                                         size_t decompressedBytesToRead,
                                                         unsigned int threadCount) {
    return readLZMAFile(archiveFile, readByteCount, decompressedBytesToRead, threadCount);
}

std::string ShapeDescriptor::utilities::generateUniqueFilenameString() {
    time_t currentTime = std::time(nullptr);
    tm convertedTime = *std::localtime(&currentTime);

    std::stringstream stream;
    // Probably should include milliseconds too
    stream << std::put_time(&convertedTime, "%Y%m%d-%H%M%S");
    return stream.str();
}