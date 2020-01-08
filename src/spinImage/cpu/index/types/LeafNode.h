#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <array>
#include "MipmapStack.h"

struct LeafNode {
    size_t length = 0;

    // Reshuffling/splitting a leaf node requires information present in
    // the mipmaps of an input image. As such we need to keep images around.
    // For space efficiency, we only keep the highest level mipmap.
    // The others can be computed based on this one.
    std::array<MipMapLevel3, 65536> imageArray;
};