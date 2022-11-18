#pragma once
#include "PointCloud.h"
#include <shapeDescriptor/cpu/types/Mesh.h>

namespace ShapeDescriptor {
    namespace gpu {
        struct Mesh {
            float* vertices_x;
            float* vertices_y;
            float* vertices_z;

            float* normals_x;
            float* normals_y;
            float* normals_z;

            size_t vertexCount;

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
            __host__ __device__
#endif
            Mesh() {
                vertexCount = 0;
            }

            ShapeDescriptor::cpu::Mesh copyToCPU();
        };

        Mesh duplicateMesh(Mesh mesh);
        void freeMesh(Mesh mesh);
    }
}
