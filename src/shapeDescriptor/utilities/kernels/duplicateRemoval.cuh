#pragma once

#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>
#include <spinImage/gpu/types/array.h>
#include <vector>

namespace SpinImage {
    namespace utilities {
        SpinImage::gpu::array<SpinImage::gpu::DeviceOrientedPoint> computeUniqueVertices(SpinImage::gpu::Mesh &mesh);
        SpinImage::gpu::array<signed long long> computeUniqueIndexMapping(SpinImage::gpu::Mesh boxScene, std::vector<SpinImage::gpu::Mesh> deviceMeshes, std::vector<size_t> *uniqueVertexCounts, size_t &totalUniqueVertexCount);
        SpinImage::gpu::array<SpinImage::gpu::DeviceOrientedPoint> applyUniqueMapping(SpinImage::gpu::Mesh boxScene, SpinImage::gpu::array<signed long long> mapping, size_t totalUniqueVertexCount);
    }
}