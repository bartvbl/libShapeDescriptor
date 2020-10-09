#pragma once

#include <cstddef>

namespace ShapeDescriptor {
    struct SampleBounds {
        size_t sampleCount;
        float areaStart;
        float areaEnd;
        size_t sampleStartIndex;
    };
}
