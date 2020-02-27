#pragma once

#include "IndexEntry.h"
#include <spinImage/cpu/types/QuiccImage.h>

struct NodeBlockEntry {
    IndexEntry indexEntry;
    QuiccImage image;
    int nextEntryIndex;
};