#pragma once

#include <shapeDescriptor/gpu/types/PointCloud.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/array.h>

namespace ShapeDescriptor {
    namespace internal {
        struct MeshSamplingBuffers {
            ShapeDescriptor::gpu::array<float> cumulativeAreaArray;
        };
    }

    namespace utilities {
        ShapeDescriptor::gpu::PointCloud sampleMesh(gpu::Mesh mesh, size_t sampleCount, size_t randomSamplingSeed, ShapeDescriptor::internal::MeshSamplingBuffers* keepComputedBuffersForExternalUse = nullptr);
    }
}