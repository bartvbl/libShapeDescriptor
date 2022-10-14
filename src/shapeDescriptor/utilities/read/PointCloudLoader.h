#pragma once

#include "shapeDescriptor/cpu/types/Mesh.h"
#include <filesystem>
#include <shapeDescriptor/cpu/types/PointCloud.h>

namespace ShapeDescriptor {
    namespace utilities {
        cpu::PointCloud loadPointCloud(std::filesystem::path src);
    }
}