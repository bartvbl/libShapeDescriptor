#pragma once

#include "IndexEntry.h"

struct NodeBlockEntry {
    IndexEntry indexEntry;
    MipMapLevel3 mipmapImage;
    int nextEntryIndex;
};