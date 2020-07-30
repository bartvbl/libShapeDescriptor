#pragma once

#include <shapeDescriptor/gpu/types/BoundingBox.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/PointCloud.h>
#include <shapeDescriptor/gpu/types/array.h>

namespace SpinImage {
    namespace utilities {
        SpinImage::gpu::BoundingBox computeBoundingBox(SpinImage::gpu::PointCloud device_pointCloud);
        SpinImage::gpu::array<unsigned int> computePointDensities(float pointDensityRadius, SpinImage::gpu::PointCloud device_pointCloud);
    }
}