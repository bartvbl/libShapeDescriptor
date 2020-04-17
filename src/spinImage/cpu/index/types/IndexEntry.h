#pragma once

#include <cstddef>
#include <spinImage/cpu/index/types/Index.h>
#include <cassert>
#include <iostream>

struct IndexEntry {
    // To save space, we only store the index of the file where the entry originated from.
    // This is translated to a full file path based on the main file list in Index.
    IndexFileID fileIndex;

    // Within the object, this is the image index that this bucket entry refers to.
    IndexImageID imageIndex;

    unsigned short remainingPixelCount;

    IndexEntry(IndexFileID fileIndex, IndexImageID imageIndex, unsigned short remainingPixels) :
        fileIndex(fileIndex),
        imageIndex(imageIndex),
        remainingPixelCount(remainingPixels) {}

    // Default constructor to allow std::vector resizing
    IndexEntry() : fileIndex(0), imageIndex(0), remainingPixelCount(0) {}
};

bool indexEntryComparator(const IndexEntry& lhs, const IndexEntry& rhs) {
    if(lhs.remainingPixelCount != rhs.remainingPixelCount) {
        return lhs.remainingPixelCount < rhs.remainingPixelCount;
    }

    if(lhs.fileIndex != rhs.fileIndex) {
        return lhs.fileIndex < rhs.fileIndex;
    }

    if(lhs.imageIndex != rhs.imageIndex) {
        return lhs.imageIndex < rhs.imageIndex;
    }
    return false;
}