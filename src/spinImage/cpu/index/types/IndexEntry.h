#pragma once

#include <cstddef>

struct IndexEntry {
    // To save space, we only store the index of the file where the entry originated from.
    // This is translated to a full file path based on the main file list in Index.
    const IndexFileID fileIndex;

    // Within the object, this is the image index that this bucket entry refers to.
    const IndexImageID imageIndex;

    IndexEntry(IndexFileID fileIndex, IndexImageID imageIndex) :
        fileIndex(fileIndex),
        imageIndex(imageIndex) {}
};