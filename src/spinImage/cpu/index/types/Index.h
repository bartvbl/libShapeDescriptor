#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <map>
#include <spinImage/libraryBuildSettings.h>
#include "NodeBlock.h"
#include <experimental/filesystem>

// Whole index section is built around images of size 64x64!
static_assert(spinImageWidthPixels == 64, "The Index part of the library assumes images are 64x64. Support for alternate image sizes must be added explicitly!");

#define INDEX_VERSION 1

// The Index struct is the struct that is shared around an application that wants to use the 'database'.
// As such it should not really contain any "temporary data" fields that are used during its construction.
struct Index {

    const std::experimental::filesystem::path indexDirectory;

    // Source file registry
    // This is done through a pointer so that accidentally creating a copy of the Index struct is not a problem
    // Otherwise, the contents of the entire vector would be copied, which can be quite large.
    const std::vector<std::experimental::filesystem::path>* indexedFileList;

    NodeBlock* rootNode;

    Index(std::experimental::filesystem::path &indexedDirectory,
          std::vector<std::experimental::filesystem::path>* indexedFiles,
          NodeBlock* root) :
            indexDirectory(indexedDirectory),
            indexedFileList(indexedFiles),
            rootNode(root) { }
};