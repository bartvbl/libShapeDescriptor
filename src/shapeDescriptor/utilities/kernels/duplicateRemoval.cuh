#pragma once

#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/common/OrientedPoint.h>
#include <shapeDescriptor/gpu/types/array.h>
#include <vector>
#include "../../common/types/methods/QUICCIDescriptor.h"

namespace ShapeDescriptor {
    namespace utilities {
        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> computeUniqueVertices(ShapeDescriptor::gpu::Mesh &mesh);

        ShapeDescriptor::gpu::array<signed long long> computeUniqueIndexMapping(ShapeDescriptor::gpu::Mesh boxScene, std::vector<ShapeDescriptor::gpu::Mesh> deviceMeshes, std::vector<size_t> *uniqueVertexCounts, size_t &totalUniqueVertexCount);
        ShapeDescriptor::gpu::array<signed long long> computeUniqueIndexMapping(ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors);

        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> applyUniqueMapping(ShapeDescriptor::gpu::Mesh boxScene, ShapeDescriptor::gpu::array<signed long long> mapping, size_t totalUniqueVertexCount);
    }
}