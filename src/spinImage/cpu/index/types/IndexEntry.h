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

    bool operator< (const IndexEntry& rhs) {
        if(remainingPixelCount != rhs.remainingPixelCount) {
            return remainingPixelCount < rhs.remainingPixelCount;
        }

        if(fileIndex != rhs.fileIndex) {
            return fileIndex < rhs.fileIndex;
        }

        if(imageIndex != rhs.imageIndex) {
            return imageIndex < rhs.imageIndex;
        }
        return false;
    }
};

