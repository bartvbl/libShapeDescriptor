#include <iomanip>
#include <fstream>
#include <cassert>
#include <spinImage/utilities/fileutils.h>
#include "IndexIO.h"

const size_t headerSize =
        // Number of image entries
        sizeof(unsigned int);

const size_t blockStructSize =
        // Leaf vs intermediate node bool array
        (NODES_PER_BLOCK / 8) +
        // Entries per node
        NODES_PER_BLOCK * sizeof(unsigned short);

NodeBlock *index::io::loadNodeBlock(const std::string &blockID, const std::experimental::filesystem::path &indexRootDirectory) {
    return nullptr;
}

void index::io::writeNodeBlock(const NodeBlock *block, const std::experimental::filesystem::path &indexRootDirectory) {
    int totalIndexEntryCount = 0;
    for(int i = 0; i < NODES_PER_BLOCK; i++) {
        totalIndexEntryCount += block->leafNodeContentsLength.at(i);
    }

    size_t entryListSize = totalIndexEntryCount * (sizeof(IndexEntry) + sizeof(MipMapLevel3));
    size_t outputBufferSize = headerSize + blockStructSize + entryListSize;

    char* outputBuffer = new char[outputBufferSize];

    

    std::experimental::filesystem::path nodeBlockFilePath = indexRootDirectory / block->identifier / "block.dat";
    std::experimental::filesystem::create_directories(nodeBlockFilePath.parent_path());

    SpinImage::utilities::writeCompressedFile(outputBuffer, outputBufferSize, nodeBlockFilePath);

    delete[] outputBuffer;
}

