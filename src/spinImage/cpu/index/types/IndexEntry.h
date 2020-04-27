#pragma once

#include <cstddef>
#include <spinImage/cpu/index/types/Index.h>

struct IndexEntry {
    // To save space, we only store the index of the file where the entry originated from.
    // This is translated to a full file path based on the main file list in Index.
    IndexFileID fileIndex;

    // Within the object, this is the image index that this bucket entry refers to.
    IndexImageID imageIndex;

    IndexEntry(IndexFileID fileIndex, IndexImageID imageIndex) :
        fileIndex(fileIndex),
        imageIndex(imageIndex) {}

    // Default constructor to allow std::vector resizing
    IndexEntry() : fileIndex(0), imageIndex(0) {}

    bool operator< (const IndexEntry& rhs) const {
        if(fileIndex != rhs.fileIndex) {
            return fileIndex < rhs.fileIndex;
        }

        if(imageIndex != rhs.imageIndex) {
            return imageIndex < rhs.imageIndex;
        }

        return false;
    }

    bool operator==(const IndexEntry& other) const {
        return fileIndex == other.fileIndex && imageIndex == other.imageIndex;
    }
};