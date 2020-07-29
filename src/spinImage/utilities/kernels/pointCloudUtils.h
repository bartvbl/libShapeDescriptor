#pragma once

#include <spinImage/gpu/types/BoundingBox.h>
#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/gpu/types/PointCloud.h>
#include <spinImage/gpu/types/array.h>

namespace SpinImage {
    namespace utilities {
        SpinImage::gpu::BoundingBox computeBoundingBox(SpinImage::gpu::PointCloud device_pointCloud);
        SpinImage::gpu::array<unsigned int> computePointDensities(float pointDensityRadius, SpinImage::gpu::PointCloud device_pointCloud);
    }
}