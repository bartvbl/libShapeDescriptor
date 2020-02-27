#include "IndexIO.h"
#include <fstream>
#include <cassert>
#include <spinImage/utilities/fileutils.h>
#include <cstring>
#include <spinImage/cpu/types/QuiccImage.h>

Index SpinImage::index::io::readIndex(std::experimental::filesystem::path indexDirectory) {
    std::experimental::filesystem::path indexFilePath = indexDirectory / "index.dat";

    NodeBlock* rootNode = SpinImage::index::io::readNodeBlock("", indexDirectory);

    size_t inputBufferSize = 0;
    const char* inputBuffer = SpinImage::utilities::readCompressedFile(indexFilePath, &inputBufferSize);

    std::vector<std::experimental::filesystem::path>* fileNames = new std::vector<std::experimental::filesystem::path>();

    int indexFileVersion = *reinterpret_cast<const int*>(inputBuffer);
    assert(indexFileVersion == INDEX_VERSION);

    int indexedFileCount = *reinterpret_cast<const int*>(inputBuffer + sizeof(int));
    fileNames->reserve(indexedFileCount);

    size_t stringBufferSize = *reinterpret_cast<const size_t*>(inputBuffer + 2 * sizeof(int));

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

    for(const auto& path : *fileNames) {
        std::cout << "Path: " << path << std::endl;
    }

    return Index(indexDirectory, fileNames, *rootNode);
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



// File format:
//
// 4 bytes: Number of index entries contained in all nodes combined
// 256 / 8 bytes: bit vector representing which node is a leaf node (true) or an intermediate one (false)
// 256 unsigned shorts: number of index entries associated with each node
// [number of entries] x (IndexEntry + QuiccImage): array containing all the entries for each node,
//                                                    sorted by which node they belong to
//
// The above is compressed and stored in a separate compressed file format

const size_t headerSize = sizeof(unsigned int);
const size_t leafNodeBoolArraySize = BoolArray<NODES_PER_BLOCK>::computeArrayLength() * sizeof(unsigned int);
const size_t entryCountArraySize = NODES_PER_BLOCK * sizeof(unsigned short);
const size_t blockStructSize = leafNodeBoolArraySize + entryCountArraySize;
const size_t entrySize = (sizeof(IndexEntry) + sizeof(QuiccImage));

NodeBlock* SpinImage::index::io::readNodeBlock(const std::string &blockID, const std::experimental::filesystem::path &indexRootDirectory) {
    std::cout << "r" << std::flush;
    std::experimental::filesystem::path nodeBlockFilePath = indexRootDirectory / blockID / "block.dat";

    assert(std::experimental::filesystem::exists(nodeBlockFilePath));

    size_t fileSize;
    const char* inputBuffer = SpinImage::utilities::readCompressedFile(nodeBlockFilePath, &fileSize);

    int totalEntryCount = *(reinterpret_cast<const int*>(inputBuffer));

    NodeBlock* nodeBlock = new NodeBlock();
    nodeBlock->identifier = blockID;
    nodeBlock->leafNodeContents.resize(totalEntryCount);
    memcpy((void *) nodeBlock->childNodeIsLeafNode.data(), &inputBuffer[headerSize], leafNodeBoolArraySize);
    memcpy(nodeBlock->leafNodeContentsLength.data(), &inputBuffer[headerSize + leafNodeBoolArraySize], entryCountArraySize);

    int nextEntryIndex = 0;
    const char* entryListBasePointer = inputBuffer + headerSize + blockStructSize;
    size_t entryPointerOffset = 0;

    for(int node = 0; node < NODES_PER_BLOCK; node++) {
        if(nodeBlock->leafNodeContentsLength.at(node) != 0) {
            nodeBlock->leafNodeContentsStartIndices.at(node) = nextEntryIndex;
            for(int entry = 0; entry < nodeBlock->leafNodeContentsLength.at(node); entry++) {
                nodeBlock->leafNodeContents.at(nextEntryIndex).indexEntry =
                        *reinterpret_cast<const IndexEntry*>(entryListBasePointer + entryPointerOffset);
                nodeBlock->leafNodeContents.at(nextEntryIndex).image =
                        *reinterpret_cast<const QuiccImage*>(entryListBasePointer + entryPointerOffset + sizeof(IndexEntry));
                nodeBlock->leafNodeContents.at(nextEntryIndex).nextEntryIndex =
                        (entry + 1 == nodeBlock->leafNodeContentsLength.at(node) ? -1 : nextEntryIndex + 1);
                entryPointerOffset += entrySize;
                nextEntryIndex++;
            }
        }
    }

    delete[] inputBuffer;
    return nodeBlock;
}

void SpinImage::index::io::writeNodeBlock(const NodeBlock *block, const std::experimental::filesystem::path &indexRootDirectory) {
    std::cout << "w" << std::flush;
    int totalIndexEntryCount = 0;
    for(int i = 0; i < NODES_PER_BLOCK; i++) {
        totalIndexEntryCount += block->leafNodeContentsLength.at(i);
    }

    size_t entryListSize = totalIndexEntryCount * entrySize;
    size_t outputBufferSize = headerSize + blockStructSize + entryListSize;

    char* outputBuffer = new char[outputBufferSize];

    *reinterpret_cast<unsigned int*>(&outputBuffer[0]) = totalIndexEntryCount;
    memcpy(&outputBuffer[headerSize], block->childNodeIsLeafNode.data(), leafNodeBoolArraySize);
    memcpy(&outputBuffer[headerSize + leafNodeBoolArraySize], block->leafNodeContentsLength.data(), entryCountArraySize);
    char* entryListBasePointer = outputBuffer + headerSize + blockStructSize;
    size_t entryPointerOffset = 0;
    for(int node = 0; node < NODES_PER_BLOCK; node++) {
        int nextListEntryIndex = block->leafNodeContentsStartIndices.at(node);
        for(int entry = 0; entry < block->leafNodeContentsLength.at(node); entry++) {
            *reinterpret_cast<IndexEntry*>(entryListBasePointer + entryPointerOffset) =
                    block->leafNodeContents.at(nextListEntryIndex).indexEntry;
            *reinterpret_cast<QuiccImage*>(entryListBasePointer + entryPointerOffset + sizeof(IndexEntry)) =
                    block->leafNodeContents.at(nextListEntryIndex).image;
            entryPointerOffset += entrySize;
            nextListEntryIndex = block->leafNodeContents.at(nextListEntryIndex).nextEntryIndex;
        }
        assert(nextListEntryIndex == -1);
    }

    std::experimental::filesystem::path nodeBlockFilePath = indexRootDirectory / block->identifier / "block.dat";
    std::experimental::filesystem::create_directories(nodeBlockFilePath.parent_path());
    assert(std::experimental::filesystem::exists(nodeBlockFilePath.parent_path()));

    SpinImage::utilities::writeCompressedFile(outputBuffer, outputBufferSize, nodeBlockFilePath);

    delete[] outputBuffer;
}

