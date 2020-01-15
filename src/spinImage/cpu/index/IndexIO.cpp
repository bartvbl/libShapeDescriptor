#include <iomanip>
#include <fstream>
#include <cassert>
#include "IndexIO.h"

std::string formatFileIndex(IndexNodeID nodeID, const unsigned int nodes_per_file) {
    IndexNodeID fileID = (nodeID / nodes_per_file) + 1;
    std::stringstream ss;
    ss << std::setw(10) << std::setfill('0') << fileID;
    return ss.str();
}

std::string formatEntryIndex(IndexNodeID nodeID, const unsigned int nodes_per_file) {
    std::stringstream ss;
    ss << std::setw(10) << std::setfill('0') << nodeID;
    return ss.str();
}

NodeBlock *index::io::loadNodeBlock(const std::string &blockID, const std::experimental::filesystem::path &indexRootDirectory) {
    return nullptr;
}

void index::io::writeNodeBlock(const NodeBlock *block, const std::experimental::filesystem::path &indexRootDirectory) {

}

