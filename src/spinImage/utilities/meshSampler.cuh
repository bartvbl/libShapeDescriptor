#pragma once

#include <spinImage/gpu/types/GPUPointCloud.h>
#include <spinImage/gpu/types/DeviceMesh.h>

namespace SpinImage {
    namespace internal {
        struct MeshSamplingBuffers {
            array<float> cumulativeAreaArray;
        };
    }

    namespace utilities {
        SpinImage::GPUPointCloud sampleMesh(DeviceMesh mesh, size_t sampleCount, size_t randomSamplingSeed, SpinImage::internal::MeshSamplingBuffers* keepComputedBuffersForExternalUse = nullptr);
    }
}