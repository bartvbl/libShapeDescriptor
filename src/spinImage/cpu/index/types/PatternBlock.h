#pragma once

#include <array>
#include <spinImage/cpu/types/BoolArray.h>
#include <spinImage/cpu/index/types/IndexEntry.h>
#include <mutex>
#include <set>
#include "PatternBlockEntry.h"

struct PatternBlock {
    std::string identifier;
    BoolArray<spinImageWidthPixels * spinImageWidthPixels> existingChildPatterns = {false};
    QuiccImage pattern;
    std::set<PatternNodeEntry> imagesContainingPattern;
    std::vector<QuiccImage> childPatterns;

    PatternBlock() {}
};