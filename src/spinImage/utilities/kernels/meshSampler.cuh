#pragma once

#include <spinImage/gpu/types/PointCloud.h>
#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/gpu/types/array.h>

namespace SpinImage {
    namespace internal {
        struct MeshSamplingBuffers {
            SpinImage::gpu::array<float> cumulativeAreaArray;
        };
    }

    namespace utilities {
        SpinImage::gpu::PointCloud sampleMesh(gpu::Mesh mesh, size_t sampleCount, size_t randomSamplingSeed, SpinImage::internal::MeshSamplingBuffers* keepComputedBuffersForExternalUse = nullptr);
    }
}