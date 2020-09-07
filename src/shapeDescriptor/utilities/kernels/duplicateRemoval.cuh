#pragma once

#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/common/OrientedPoint.h>
#include <shapeDescriptor/gpu/types/array.h>
#include <vector>

namespace ShapeDescriptor {
    namespace utilities {
        ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::OrientedPoint> computeUniqueVertices(ShapeDescriptor::gpu::Mesh &mesh);
        ShapeDescriptor::gpu::array<signed long long> computeUniqueIndexMapping(ShapeDescriptor::gpu::Mesh boxScene, std::vector<ShapeDescriptor::gpu::Mesh> deviceMeshes, std::vector<size_t> *uniqueVertexCounts, size_t &totalUniqueVertexCount);
        ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::OrientedPoint> applyUniqueMapping(ShapeDescriptor::gpu::Mesh boxScene, ShapeDescriptor::gpu::array<signed long long> mapping, size_t totalUniqueVertexCount);
    }
}