#pragma once

#include <filesystem>
#include <shapeDescriptor/cpu/types/PointCloud.h>

namespace ShapeDescriptor {
    namespace utilities {
        void writeXYZ(std::filesystem::path destination, ShapeDescriptor::cpu::PointCloud pointCloud);
    }
}
