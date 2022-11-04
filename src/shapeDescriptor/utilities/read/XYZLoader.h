#pragma once

#include <filesystem>
#include <shapeDescriptor/cpu/types/PointCloud.h>

namespace ShapeDescriptor {
    namespace utilities {
        cpu::PointCloud loadXYZ(std::filesystem::path src, bool readNormals = false, bool readColours = false);
    }
}
