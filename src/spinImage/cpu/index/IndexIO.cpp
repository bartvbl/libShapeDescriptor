#include "IndexIO.h"
#include <fstream>
#include <cassert>
#include <spinImage/utilities/fileutils.h>
#include <cstring>

// File format:
//
// 4 bytes: Number of index entries contained in all nodes combined
// 256 / 8 bytes: bit vector representing which node is a leaf node (true) or an intermediate one (false)
// 256 unsigned shorts: number of index entries associated with each node
// [number of entries] x (IndexEntry + MipmapLevel3): array containing all the entries for each node,
//                                                    sorted by which node they belong to
//
// The above is compressed and stored in a separate compressed file format

const size_t headerSize = sizeof(unsigned int);
const size_t leafNodeBoolArraySize = BoolArray<NODES_PER_BLOCK>::computeArrayLength() * sizeof(unsigned int);
const size_t entryCountArraySize = NODES_PER_BLOCK * sizeof(unsigned short);
const size_t blockStructSize = leafNodeBoolArraySize + entryCountArraySize;
const size_t entrySize = (sizeof(IndexEntry) + sizeof(MipMapLevel3));

NodeBlock* SpinImage::index::io::loadNodeBlock(const std::string &blockID, const std::experimental::filesystem::path &indexRootDirectory) {
    std::experimental::filesystem::path nodeBlockFilePath = indexRootDirectory / blockID / "block.dat";

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
                nodeBlock->leafNodeContents.at(nextEntryIndex).mipmapImage =
                        *reinterpret_cast<const MipMapLevel3*>(entryListBasePointer + entryPointerOffset + sizeof(IndexEntry));
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
    std::cout << "Writing block " << block->identifier << ".." << std::endl;
    int totalIndexEntryCount = 0;
    for(int i = 0; i < NODES_PER_BLOCK; i++) {
        totalIndexEntryCount += block->leafNodeContentsLength.at(i);
    }

    size_t entryListSize = totalIndexEntryCount * entrySize;
    size_t outputBufferSize = headerSize + blockStructSize + entryListSize;

    char* outputBuffer = new char[outputBufferSize];

    *reinterpret_cast<unsigned int*>(outputBuffer[0]) = totalIndexEntryCount;
    memcpy(&outputBuffer[headerSize], block->childNodeIsLeafNode.data(), leafNodeBoolArraySize);
    memcpy(&outputBuffer[headerSize + leafNodeBoolArraySize], block->leafNodeContentsLength.data(), entryCountArraySize);
    char* entryListBasePointer = outputBuffer + headerSize + blockStructSize;
    size_t entryPointerOffset = 0;
    for(int node = 0; node < NODES_PER_BLOCK; node++) {
        int nextListEntryIndex = block->leafNodeContentsStartIndices.at(node);
        for(int entry = 0; entry < block->leafNodeContentsLength.at(node); entry++) {
            *reinterpret_cast<IndexEntry*>(entryListBasePointer + entryPointerOffset) =
                    block->leafNodeContents.at(nextListEntryIndex).indexEntry;
            *reinterpret_cast<MipMapLevel3*>(entryListBasePointer + entryPointerOffset + sizeof(IndexEntry)) =
                    block->leafNodeContents.at(nextListEntryIndex).mipmapImage;
            entryPointerOffset += entrySize;
            nextListEntryIndex = block->leafNodeContents.at(nextListEntryIndex).nextEntryIndex;
        }
        assert(nextListEntryIndex == -1);
    }

    std::experimental::filesystem::path nodeBlockFilePath = indexRootDirectory / block->identifier / "block.dat";
    std::experimental::filesystem::create_directories(nodeBlockFilePath.parent_path());

    SpinImage::utilities::writeCompressedFile(outputBuffer, outputBufferSize, nodeBlockFilePath);

    delete[] outputBuffer;
}

