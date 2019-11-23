#pragma once

#include <cstddef>

struct IndexEntry {
    // To save space, we only store the index of the file where the entry originated from.
    // This is translated to a full file path based on the main file list in Index.
    size_t fileIndex;

    // Within the object, this is the image index that this bucket entry refers to.
    size_t imageIndex;
};