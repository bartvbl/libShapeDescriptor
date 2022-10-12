#pragma once

#include <shapeDescriptor/cpu/types/PointCloud.h>
#include <shapeDescriptor/gpu/types/PointCloud.h>

namespace ShapeDescriptor {
    namespace copy{
        cpu::PointCloud devicePointCloudToHost(gpu::PointCloud deviceMesh);
        gpu::PointCloud hostPointCloudToDevice(cpu::PointCloud hostMesh);
    }
}