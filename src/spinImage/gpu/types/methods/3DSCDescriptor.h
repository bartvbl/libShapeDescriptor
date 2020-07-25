#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <array>

namespace SpinImage {
    namespace gpu {
        struct ShapeContextDescriptor {
            std::array<shapeContextBinType,
                    SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT *
                    SHAPE_CONTEXT_VERTICAL_SLICE_COUNT *
                    SHAPE_CONTEXT_LAYER_COUNT> contents;
        };
    }
}

