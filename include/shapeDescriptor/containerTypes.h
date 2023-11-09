#pragma once

namespace ShapeDescriptor {
    namespace internal {
        void gpuMemsetMultibyte(char *array, size_t length, const char *value, size_t valueSize);
    }

    template<typename TYPE>
    void setValue(TYPE *array, size_t length, TYPE value) {
        CUDA_REGION(
        // A function boundary is necessary to ensure the associated GPU kernel can be called when this template is
        // included from a regular non-CUDA source file
                internal::gpuMemsetMultibyte(reinterpret_cast<char *>(array), length, reinterpret_cast<char *>(&value),
                                             sizeof(value));
        )
    }

    namespace cpu {
        template<typename TYPE>
        struct array;
        struct Mesh;
    }
    namespace gpu {
        template<typename TYPE>
        struct array;
        struct Mesh;
    }

    template<typename T>
    ShapeDescriptor::cpu::array<T> copyToCPU(ShapeDescriptor::gpu::array<T> array) {
        CUDA_REGION(
                ShapeDescriptor::cpu::array<T> hostArray;
                hostArray.length = array.length;
                hostArray.content = new T[array.length];

                checkCudaErrors(
                        cudaMemcpy(hostArray.content, array.content, array.length * sizeof(T), cudaMemcpyDeviceToHost));

                return hostArray;
        )
    }

    template<typename T>
    ShapeDescriptor::gpu::array<T> copyToGPU(ShapeDescriptor::cpu::array<T> array) {
        CUDA_REGION(
                ShapeDescriptor::gpu::array<T> deviceArray;
                deviceArray.length = array.length;

                size_t bufferSize = sizeof(T) * array.length;

                checkCudaErrors(cudaMalloc((void **) &deviceArray.content, bufferSize));
                checkCudaErrors(cudaMemcpy(deviceArray.content, array.content, bufferSize, cudaMemcpyHostToDevice));

                return deviceArray;
        )
    }


    namespace gpu {
        struct VertexList {
            float *array = nullptr;
            size_t length = 0;

            // For copying
            VertexList() {}

            VertexList(size_t length) {
                CUDA_REGION(
                        checkCudaErrors(cudaMalloc((void **) &array, 3 * length * sizeof(float)));
                        this->length = length;
                )
            }

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
            __device__ float3 at(size_t index) {
                assert(index < length);

                float3 item;
                item.x = array[index];
                item.y = array[index + length];
                item.z = array[index + 2 * length];
                return item;
            }

            __device__ void set(size_t index, float3 value) {
                assert(index < length);

                array[index] = value.x;
                array[index + length] = value.y;
                array[index + 2 * length] = value.z;
            }

#endif

            void free() {
                CUDA_REGION(checkCudaErrors(cudaFree(array));)
            }
        };


        template<typename TYPE>
        struct array {
            size_t length = 0;
            TYPE *content = nullptr;

            __host__ __device__ array() {}

            __host__ array(size_t length) {
                CUDA_REGION(
                this->length = length;
                cudaMalloc(&content, length * sizeof(TYPE));
                )
            }

            __host__ __device__ array(size_t length, TYPE *content) : length(length), content(content) {}

            __host__ ShapeDescriptor::cpu::array<TYPE> copyToCPU() {
                return ShapeDescriptor::copyToCPU<TYPE>({length, content});
            }

            __host__ void setValue(TYPE &value) {
                ShapeDescriptor::setValue<TYPE>(content, length, value);
            }

            __device__ TYPE operator[](size_t index) {
                return content[index];
            }
        };
    }

    namespace cpu {
        template<typename TYPE>
        struct array {
            size_t length;
            TYPE *content;

            array() {}

            array(size_t length)
                    : length(length),
                      content(new TYPE[length]) {}

            array(size_t length, TYPE *content)
                    : length(length),
                      content(content) {}

            ShapeDescriptor::gpu::array<TYPE> copyToGPU() {
                return ShapeDescriptor::copyToGPU<TYPE>({length, content});
            }

            void setValue(TYPE &value) {
                std::fill_n(content, length, value);
            }

            TYPE &operator[](size_t index) {
                //assert(index >= 0); // >= 0
                assert(index < length);
                return *(content + index);
            }
        };
    }


    /* -- Mesh and Point cloud types -- */

    namespace cpu {
        struct Mesh {
            ShapeDescriptor::cpu::float3 *vertices = nullptr;
            ShapeDescriptor::cpu::float3 *normals = nullptr;
            ShapeDescriptor::cpu::uchar4 *vertexColours = nullptr;

            size_t vertexCount = 0;

            Mesh() = default;

            Mesh(size_t vertCount) {
                vertices = new ShapeDescriptor::cpu::float3[vertCount];
                normals = new ShapeDescriptor::cpu::float3[vertCount];
                vertexColours = new ShapeDescriptor::cpu::uchar4[vertCount];
                vertexCount = vertCount;
            }

            ShapeDescriptor::gpu::Mesh copyToGPU();

            ShapeDescriptor::cpu::Mesh clone() const;
        };

        struct PointCloud {
            ShapeDescriptor::cpu::float3 *vertices = nullptr;
            ShapeDescriptor::cpu::float3 *normals = nullptr;
            ShapeDescriptor::cpu::uchar4 *vertexColours = nullptr;

            size_t pointCount = 0;
            bool hasVertexNormals = false;
            bool hasVertexColours = false;

            PointCloud() = default;

            PointCloud(size_t pointCount) {
                vertices = new ShapeDescriptor::cpu::float3[pointCount];
                normals = new ShapeDescriptor::cpu::float3[pointCount];
                vertexColours = new ShapeDescriptor::cpu::uchar4[pointCount];
                this->pointCount = pointCount;
            }
        };
    }

    namespace gpu {
        struct Mesh {
            float *vertices_x;
            float *vertices_y;
            float *vertices_z;

            float *normals_x;
            float *normals_y;
            float *normals_z;

            size_t vertexCount;

            Mesh() {
                vertexCount = 0;
            }

            ShapeDescriptor::cpu::Mesh copyToCPU();
        };

        Mesh duplicateMesh(Mesh mesh);

        void free(const Mesh &mesh);

        struct PointCloud {
            VertexList vertices;
            VertexList normals;
            size_t pointCount;

            PointCloud() = default;

            PointCloud(size_t pointCount) : vertices(pointCount), normals(pointCount), pointCount(pointCount) {}

            void free() {
                vertices.free();
                normals.free();
            }
        };
    }
}