#pragma once

#include <shapeDescriptor/libraryBuildSettings.h>

namespace ShapeDescriptor {
    namespace gpu {
        struct ShapeContextDescriptor {
            shapeContextBinType contents[
                    SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT *
                    SHAPE_CONTEXT_VERTICAL_SLICE_COUNT *
                    SHAPE_CONTEXT_LAYER_COUNT];
        };
    }
}

