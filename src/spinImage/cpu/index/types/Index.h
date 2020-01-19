#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <map>
#include <spinImage/libraryBuildSettings.h>
#include <experimental/filesystem>


// Due to parsing order of header files, these must be at the top, before the remaining includes
// They represent a tradeoff between the number of files/images the database is able to represent,
// relative to the amount of data it costs to store them on disk and in memory
typedef unsigned int IndexFileID;
typedef size_t IndexNodeID;
typedef unsigned int IndexImageID;

// Whole index section is built around images of size 64x64!
static_assert(spinImageWidthPixels == 64, "The Index part of the library assumes images are 64x64. Support for alternate image sizes must be added explicitly!");

#include "NodeBlock.h"


// The Index struct is the struct that is shared around an application that wants to use the 'database'.
// As such it should not really contain any "temporary data" fields that are used during its construction.
struct Index {

    const std::experimental::filesystem::path indexDirectory;

    // Source file registry
    // This is done through a pointer so that accidentally creating a copy of the Index struct is not a problem
    // Otherwise, the contents of the entire vector would be copied, which can be quite large.
    const std::vector<std::experimental::filesystem::path>* indexedFileList;

    const IndexNodeID indexNodeCount;
    const IndexNodeID bucketNodeCount;

    const NodeBlock rootNode;

    Index(std::experimental::filesystem::path &indexedDirectory,
          std::vector<std::experimental::filesystem::path>* indexedFiles,
          NodeBlock &root,
          IndexNodeID indexNodeCount,
          IndexNodeID bucketNodeCount) :
            indexDirectory(indexedDirectory),
            indexedFileList(indexedFiles),
            rootNode(root),
            indexNodeCount(indexNodeCount),
            bucketNodeCount(bucketNodeCount) { }
};