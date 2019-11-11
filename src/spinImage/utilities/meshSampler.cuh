#pragma once

#include <spinImage/gpu/types/PointCloud.h>
#include <spinImage/gpu/types/Mesh.h>

namespace SpinImage {
    namespace internal {
        struct MeshSamplingBuffers {
            array<float> cumulativeAreaArray;
        };
    }

    namespace utilities {
        gpu::PointCloud sampleMesh(gpu::Mesh mesh, size_t sampleCount, size_t randomSamplingSeed, SpinImage::internal::MeshSamplingBuffers* keepComputedBuffersForExternalUse = nullptr);
    }
}