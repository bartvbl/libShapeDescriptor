#pragma once

#include <cstddef>

namespace SpinImage {
    struct SampleBounds {
        size_t sampleCount;
        float areaStart;
        float areaEnd;
        size_t sampleStartIndex;
    };
}
