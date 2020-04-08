#pragma once

#include <spinImage/cpu/types/QuiccImage.h>
#include <vector>

namespace SpinImage {
    namespace index {
        namespace pattern {
            bool findNext(
                    QuiccImage &image, QuiccImage &foundPattern,
                    unsigned int &foundPatternSize,
                    unsigned int &startRow, unsigned int &startColumn,
                    std::vector<std::pair<unsigned short, unsigned short>> &floodFillBuffer);
        }
    }
}