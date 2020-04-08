#pragma once

#include "IndexEntry.h"
#include <spinImage/cpu/types/QuiccImage.h>

struct PatternBlockEntry {
    IndexEntry indexEntry;
    unsigned short remainingPixelCount;

    PatternBlockEntry(const IndexEntry entry, unsigned short pixelsRemaining)
        : indexEntry(entry),
          remainingPixelCount(pixelsRemaining) {}
    PatternBlockEntry() {}
};