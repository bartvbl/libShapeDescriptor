#pragma once

#include <shapeDescriptor/cpu/types/PointCloud.h>
#include <shapeDescriptor/gpu/types/PointCloud.h>

namespace ShapeDescriptor {
    namespace free {
        void pointCloud(ShapeDescriptor::cpu::PointCloud &cloudToFree);
        void pointCloud(ShapeDescriptor::gpu::PointCloud &cloudToFree);
    }
}