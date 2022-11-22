#pragma once

#include <shapeDescriptor/gpu/types/PointCloud.h>
#include <shapeDescriptor/cpu/types/PointCloud.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/array.h>

namespace ShapeDescriptor {
    namespace utilities {
        ShapeDescriptor::gpu::PointCloud sampleMesh(gpu::Mesh mesh, size_t sampleCount, size_t randomSamplingSeed);
        ShapeDescriptor::cpu::PointCloud sampleMesh(cpu::Mesh mesh, size_t sampleCount, size_t randomSamplingSeed);
    }
}