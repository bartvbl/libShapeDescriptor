#pragma once
#include <cassert>
#include <cstddef>
#include <vector>

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
            __device__ float3 at(size_t index) const {
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

            ShapeDescriptor::gpu::array<TYPE> copyToGPU() const {
                return ShapeDescriptor::copyToGPU<TYPE>({length, content});
            }

            void setValue(TYPE &value) {
                std::fill_n(content, length, value);
            }

            ShapeDescriptor::cpu::array<TYPE> clone() {
                ShapeDescriptor::cpu::array<TYPE> copiedArray(length);
                std::copy(content, content + length, copiedArray.content);
                return copiedArray;
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
        struct MeshTexture {
            std::vector<uint8_t> textureData;

            uint8_t bytesPerPixel = 3;
            uint8_t channelsPerPixel = 3;
            uint32_t widthPixels = 0;
            uint32_t heightPixels = 0;
        };

        class TextureCollection {
            std::vector<MeshTexture> textures;

            template<typename Pixel>
            Pixel readPixel(uint32_t x, uint32_t y, const MeshTexture& texture) {
                uint32_t coordinateX = std::min(x, texture.widthPixels - 1);
                uint32_t coordinateY = std::min(y, texture.heightPixels - 1);
                uint32_t index = coordinateY * texture.widthPixels + coordinateX;
                Pixel output;
                memcpy(&output, texture.textureData.data() + (index * texture.bytesPerPixel), sizeof(Pixel));
                return output;
            }

        public:
            template<typename Pixel>
            Pixel getPixelAt(ShapeDescriptor::cpu::float2 textureCoordinate, uint8_t textureIndex) {
                const MeshTexture& texture = textures.at(textureIndex);
                if(sizeof(Pixel) != texture.bytesPerPixel) {
                    throw std::runtime_error("Pixel size mismatch! Attempted to read pixel with size of " + std::to_string(sizeof(Pixel)) + " bytes, but texture has " + std::to_string(texture.bytesPerPixel) + " bytes per pixel.");
                }
                float exactPixelX = std::min<float>(1, std::max<float>(textureCoordinate.x, 0)) * float(texture.widthPixels);
                float exactPixelY = std::min<float>(1, std::max<float>(textureCoordinate.y, 0)) * float(texture.heightPixels);
                uint32_t baseX = uint32_t(exactPixelX);
                uint32_t baseY = uint32_t(exactPixelY);
                float distanceInPixelX = float(baseX) - exactPixelX;
                float distanceInPixelY = float(baseY) - exactPixelY;

                Pixel bottomLeftPixel = readPixel<Pixel>(baseX, baseY, texture);
                Pixel bottomRightPixel = readPixel<Pixel>(baseX + 1, baseY, texture);
                Pixel topRightPixel = readPixel<Pixel>(baseX + 1, baseY + 1, texture);
                Pixel topLeftPixel = readPixel<Pixel>(baseX, baseY + 1, texture);

                Pixel bilinearBottomX = (1 - distanceInPixelX) * bottomLeftPixel + distanceInPixelX * bottomRightPixel;
                Pixel bilinearTopX = (1 - distanceInPixelX) * topLeftPixel + distanceInPixelX * topRightPixel;
                Pixel interpolatedPixel = (1 - distanceInPixelY) * bilinearBottomX + distanceInPixelY * bilinearTopX;

                return interpolatedPixel;
            }
        };

        struct Mesh {
            ShapeDescriptor::cpu::float3* vertices = nullptr;
            ShapeDescriptor::cpu::float3* normals = nullptr;
            ShapeDescriptor::cpu::uchar4* vertexColours = nullptr;
            ShapeDescriptor::cpu::float2* textureCoordinates = nullptr;

            // Which texture should be sampled for each triangle in the mesh
            uint8_t* triangleTextureIDs = nullptr;

            bool hasTexture = false;
            TextureCollection texture;

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

            ShapeDescriptor::cpu::PointCloud clone() const;
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