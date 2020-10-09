#pragma once

#include <shapeDescriptor/common/types/BoundingBox.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/PointCloud.h>
#include <shapeDescriptor/gpu/types/array.h>

namespace ShapeDescriptor {
    namespace utilities {
        ShapeDescriptor::BoundingBox computeBoundingBox(ShapeDescriptor::gpu::PointCloud device_pointCloud);
        ShapeDescriptor::gpu::array<unsigned int> computePointDensities(float pointDensityRadius, ShapeDescriptor::gpu::PointCloud device_pointCloud);
    }
}