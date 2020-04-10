#include "IndexIO.h"
#include <fstream>
#include <cassert>
#include <cstring>
#include <iostream>
#include <spinImage/cpu/types/QuiccImage.h>
#include <spinImage/utilities/compression/CompressedFileWriter.h>
#include <spinImage/utilities/compression/CompressedFileReader.h>

Index SpinImage::index::io::readIndex(std::experimental::filesystem::path indexDirectory) {
    std::experimental::filesystem::path indexFilePath = indexDirectory / "index.dat";

    size_t inputBufferSize = 0;
    const char* inputBuffer = SpinImage::utilities::readCompressedFile(indexFilePath, &inputBufferSize, true);

    std::vector<std::experimental::filesystem::path>* fileNames = new std::vector<std::experimental::filesystem::path>();

    int indexFileVersion = *reinterpret_cast<const int*>(inputBuffer);
    assert(indexFileVersion == INDEX_VERSION);

    int indexedFileCount = *reinterpret_cast<const int*>(inputBuffer + sizeof(int));
    fileNames->reserve(indexedFileCount);

    const char* nextStringBufferEntry = inputBuffer + 2 * sizeof(int) + sizeof(size_t);

    std::string tempPathString;
    // Reserve 1MB
    tempPathString.reserve(1024*1024);
    for(unsigned int entry = 0; entry < indexedFileCount; entry++) {
        int pathNameSize = *reinterpret_cast<const int*>(nextStringBufferEntry);
        nextStringBufferEntry += sizeof(int);
        tempPathString.assign(nextStringBufferEntry, pathNameSize);
        // Add null terminator
        fileNames->push_back(tempPathString);
        nextStringBufferEntry += pathNameSize;
    }

    delete[] inputBuffer;

    std::cout << "Index has " << fileNames->size() << " files." << std::endl;

    return Index(indexDirectory, fileNames);
}

void SpinImage::index::io::writeIndex(const Index& index, std::experimental::filesystem::path indexDirectory) {
    std::experimental::filesystem::path indexFilePath = indexDirectory / "index.dat";

    size_t stringArraySize = 0;
    for(const std::experimental::filesystem::path& filename : *index.indexedFileList) {
        stringArraySize += sizeof(int);
        stringArraySize += std::experimental::filesystem::absolute(filename).string().size() * sizeof(std::string::value_type);
    }

    size_t fileHeaderSize = sizeof(int) + sizeof(int) + sizeof(size_t);
    size_t indexFileSize = fileHeaderSize + stringArraySize;

    char* outputFileBuffer = new char[indexFileSize];

    *reinterpret_cast<int*>(outputFileBuffer) = int(INDEX_VERSION);
    *reinterpret_cast<int*>(outputFileBuffer + sizeof(int)) = int((*index.indexedFileList).size());
    *reinterpret_cast<size_t*>(outputFileBuffer + 2 * sizeof(int)) = stringArraySize;

    char* nextStringEntryPointer = outputFileBuffer + fileHeaderSize;

    for(const std::experimental::filesystem::path& filename : *index.indexedFileList) {
        std::string pathString = std::experimental::filesystem::absolute(filename).string();
        size_t pathSizeInBytes = pathString.size() * sizeof(std::string::value_type);
        *reinterpret_cast<int*>(nextStringEntryPointer) = int(pathSizeInBytes);
        nextStringEntryPointer += sizeof(int);
        std::copy(pathString.begin(), pathString.end(), nextStringEntryPointer);
        nextStringEntryPointer += pathSizeInBytes;
    }

    assert((nextStringEntryPointer - outputFileBuffer - fileHeaderSize) == stringArraySize);

    SpinImage::utilities::writeCompressedFile(outputFileBuffer, indexFileSize, indexFilePath);

    delete[] outputFileBuffer;
}
