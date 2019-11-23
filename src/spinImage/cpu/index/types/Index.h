#pragma once


#include <string>
#include <vector>
#include <map>
#include "IndexNode.h"
#include <spinImage/libraryBuildSettings.h>

// Whole index section is built around images of size 64x64!
static_assert(spinImageWidthPixels == 64);

struct Index {
    // Source file registry
    std::string quicciImageDumpDirectory;
    std::vector<std::string> indexedFileList;

    // Root of index
    std::vector<IndexNode> rootNodes;
};