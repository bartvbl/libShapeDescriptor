#include "duplicateRemoval.cuh"

#include <vector>
#include <nvidia/helper_cuda.h>
#include <cassert>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/utilities/copy/array.h>
#include <shapeDescriptor/utilities/free/array.h>

__host__ __device__ __inline__ size_t roundSizeToNearestCacheLine(size_t sizeInBytes) {
    return (sizeInBytes + 127u) & ~((size_t) 127);
}


__global__ void detectDuplicates(ShapeDescriptor::gpu::Mesh mesh, bool* isDuplicate) {
    size_t vertexIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(vertexIndex >= mesh.vertexCount) {
        return;
    }

    float3 vertex = make_float3(
            mesh.vertices_x[vertexIndex],
            mesh.vertices_y[vertexIndex],
            mesh.vertices_z[vertexIndex]);
    float3 normal = make_float3(
            mesh.normals_x[vertexIndex],
            mesh.normals_y[vertexIndex],
            mesh.normals_z[vertexIndex]);

    for(size_t i = 0; i < vertexIndex; i++) {
        float3 otherVertex = make_float3(
                mesh.vertices_x[i],
                mesh.vertices_y[i],
                mesh.vertices_z[i]);
        float3 otherNormal = make_float3(
                mesh.normals_x[i],
                mesh.normals_y[i],
                mesh.normals_z[i]);

        // We're looking for exact matches here. Given that vertex duplications should
        // yield equivalent vertex coordinates, testing floating point numbers for
        // exact equivalence is warranted.
        if( vertex.x == otherVertex.x &&
            vertex.y == otherVertex.y &&
            vertex.z == otherVertex.z &&
            normal.x == otherNormal.x &&
            normal.y == otherNormal.y &&
            normal.z == otherNormal.z) {

            isDuplicate[vertexIndex] = true;
            return;
        }
    }

    isDuplicate[vertexIndex] = false;
}

__global__ void computeTargetIndices(ShapeDescriptor::gpu::array<signed long long> targetIndices, bool* duplicateVertices, size_t vertexCount) {
    size_t vertexIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(vertexIndex >= vertexCount) {
        return;
    }

    // The value of -1 indicates that the vertex is a duplicate of another one
    // and should therefore be discarded
    signed long long targetIndex = -1;

    bool isDuplicate = duplicateVertices[vertexIndex];

    if(!isDuplicate) {
        for(size_t i = 0; i < vertexIndex; i++) {
            // If it is a duplicate, it will get removed
            // Otherwise, it'll be added in front of the current entry
            targetIndex += duplicateVertices[i] ? 0 : 1;
        }
    }

    targetIndices.content[vertexIndex] = targetIndex;
}

ShapeDescriptor::gpu::array<signed long long> ShapeDescriptor::utilities::computeUniqueIndexMapping(ShapeDescriptor::gpu::Mesh boxScene, std::vector<ShapeDescriptor::gpu::Mesh> deviceMeshes, std::vector<size_t> *uniqueVertexCounts, size_t &totalUniqueVertexCount) {
    size_t sceneVertexCount = boxScene.vertexCount;

    bool* device_duplicateVertices;
    checkCudaErrors(cudaMalloc(&device_duplicateVertices, sceneVertexCount * sizeof(bool)));
    detectDuplicates<<<(boxScene.vertexCount / 256) + 1, 256>>>(boxScene, device_duplicateVertices);
    checkCudaErrors(cudaDeviceSynchronize());

    bool* temp_duplicateVertices = new bool[sceneVertexCount];
    checkCudaErrors(cudaMemcpy(temp_duplicateVertices, device_duplicateVertices, boxScene.vertexCount * sizeof(bool), cudaMemcpyDeviceToHost));

    //std::fstream tempOutFile("DEBUG_duplicates_" + std::to_string(boxScene.vertexCount) + ".txt", std::ios::out);

    ShapeDescriptor::cpu::Mesh temp_host_boxScene = ShapeDescriptor::copy::deviceMeshToHost(boxScene);

    size_t baseIndex = 0;
    totalUniqueVertexCount = 0;
    for(auto mesh : deviceMeshes) {
        //tempOutFile << "=== MESH " << mesh.vertexCount << " ===" << std::endl << std::endl;
        size_t meshUniqueVertexCount = 0;
        for(size_t i = 0; i < mesh.vertexCount; i++) {
            // Check if the vertex is unique
            if(temp_duplicateVertices[baseIndex + i] == false) {
                totalUniqueVertexCount++;
                meshUniqueVertexCount++;
            }
            ShapeDescriptor::cpu::float3 vertex = temp_host_boxScene.vertices[baseIndex + i];
            ShapeDescriptor::cpu::float3 normal = temp_host_boxScene.normals[baseIndex + i];

            std::string vertexString = vertex.to_string();
            std::string normalString = normal.to_string();

            /*tempOutFile << vertexString;
            for(int i = vertexString.size(); i < 40; i++) {
                tempOutFile << " ";
            }
            tempOutFile << normalString;
            for(int i = normalString.size(); i < 40; i++) {
                tempOutFile << " ";
            }
            tempOutFile << ": " << temp_duplicateVertices[baseIndex + i] << std::endl;*/
        }
        baseIndex += meshUniqueVertexCount;
        uniqueVertexCounts->push_back(meshUniqueVertexCount);
    }

    ShapeDescriptor::free::mesh(temp_host_boxScene);
    delete[] temp_duplicateVertices;

    ShapeDescriptor::gpu::array<signed long long> device_uniqueIndexMapping;
    device_uniqueIndexMapping.length = boxScene.vertexCount;
    checkCudaErrors(cudaMalloc(&device_uniqueIndexMapping.content, boxScene.vertexCount * sizeof(signed long long)));
    computeTargetIndices<<<(boxScene.vertexCount / 256) + 1, 256>>>(device_uniqueIndexMapping, device_duplicateVertices, boxScene.vertexCount);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(device_duplicateVertices));

    return device_uniqueIndexMapping;
}

__global__ void mapVertices(ShapeDescriptor::gpu::Mesh boxScene, ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> origins, ShapeDescriptor::gpu::array<signed long long> mapping) {
    size_t vertexIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(vertexIndex >= boxScene.vertexCount) {
        return;
    }

    signed long long targetIndex = mapping.content[vertexIndex];

    if(targetIndex != -1 && targetIndex < origins.length) {
        float3 vertex = make_float3(
                boxScene.vertices_x[vertexIndex],
                boxScene.vertices_y[vertexIndex],
                boxScene.vertices_z[vertexIndex]);
        float3 normal = make_float3(
                boxScene.normals_x[vertexIndex],
                boxScene.normals_y[vertexIndex],
                boxScene.normals_z[vertexIndex]);

        ShapeDescriptor::OrientedPoint origin;
        origin.vertex = vertex;
        origin.normal = normal;

        origins.content[targetIndex] = origin;
    }
}

ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> ShapeDescriptor::utilities::applyUniqueMapping(ShapeDescriptor::gpu::Mesh boxScene, ShapeDescriptor::gpu::array<signed long long> device_mapping, size_t totalUniqueVertexCount) {
    assert(boxScene.vertexCount == device_mapping.length);

    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_origins;
    device_origins.length = totalUniqueVertexCount;
    checkCudaErrors(cudaMalloc(&device_origins.content, totalUniqueVertexCount * sizeof(ShapeDescriptor::OrientedPoint)));

    mapVertices<<<(boxScene.vertexCount / 256) + 1, 256>>>(boxScene, device_origins, device_mapping);
    checkCudaErrors(cudaDeviceSynchronize());

    return device_origins;
}

ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> ShapeDescriptor::utilities::computeUniqueVertices(ShapeDescriptor::gpu::Mesh &mesh) {
    std::vector<ShapeDescriptor::gpu::Mesh> deviceMeshes;
    deviceMeshes.push_back(mesh);
    std::vector<size_t> vertexCounts;
    size_t totalUniqueVertexCount;
    ShapeDescriptor::gpu::array<signed long long> device_mapping = computeUniqueIndexMapping(mesh, deviceMeshes, &vertexCounts, totalUniqueVertexCount);
    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_origins = applyUniqueMapping(mesh, device_mapping, totalUniqueVertexCount);
    checkCudaErrors(cudaFree(device_mapping.content));
    return device_origins;
}

__device__ bool isQuicciPairEquivalent(
        ShapeDescriptor::QUICCIDescriptor* imageA,
        ShapeDescriptor::QUICCIDescriptor* imageB) {
    for(unsigned int i = threadIdx.x % 32; i < UINTS_PER_QUICCI; i += 32) {
        bool isEqual = imageA->contents[i] == imageB->contents[i];

        if(__ballot_sync(0xFFFFFFFF, isEqual) != 0xFFFFFFFF) {
            return false;
        }
    }
    return true;
}

__global__ void computeDuplicateQUICCIMapping(
        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
        ShapeDescriptor::gpu::array<signed long long> mappingIndices,
        unsigned int* uniqueElementCount) {
#define IMAGE_INDEX blockIdx.x

    __shared__ ShapeDescriptor::QUICCIDescriptor descriptor;

    for(unsigned int i = threadIdx.x; i < ShapeDescriptor::QUICCIDescriptorLength; i += blockDim.x) {
        descriptor.contents[i] = descriptors.content[IMAGE_INDEX].contents[i];
    }

    __syncthreads();

    for(size_t i = 0; i < IMAGE_INDEX; i++) {
        if(isQuicciPairEquivalent(&descriptor, &descriptors.content[i])) {
            mappingIndices.content[IMAGE_INDEX] = i;
            return;
        }
    }

    // No duplicate found, mark image as unique
    mappingIndices.content[IMAGE_INDEX] = -1;

    if(threadIdx.x == 0) {
        atomicAdd(uniqueElementCount, 1);
    }
}


ShapeDescriptor::utilities::DuplicateMapping ShapeDescriptor::utilities::computeUniqueIndexMapping(ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors) {
    ShapeDescriptor::utilities::DuplicateMapping mapping;
    mapping.mappedIndices = ShapeDescriptor::gpu::array<signed long long>(descriptors.length);
    unsigned int* device_uniqueElementCount;
    checkCudaErrors(cudaMalloc(&device_uniqueElementCount, sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(device_uniqueElementCount, 0, sizeof(unsigned int)));

    computeDuplicateQUICCIMapping<<<descriptors.length, 32>>>(descriptors, mapping.mappedIndices, device_uniqueElementCount);

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(&mapping.uniqueElementCount, device_uniqueElementCount, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    return mapping;
}

__global__ void mapImages(
        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> sourceDescriptors,
        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> targetDescriptors,
        ShapeDescriptor::gpu::array<size_t> targetIndices) {
    // Copies a single image per block
    size_t targetIndex = targetIndices.content[blockIdx.x];
    for(unsigned int i = threadIdx.x; i < UINTS_PER_QUICCI; i += blockDim.x) {
        targetDescriptors.content[targetIndex].contents[i] = sourceDescriptors.content[blockIdx.x].contents[i];
    }
}

ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> ShapeDescriptor::utilities::applyUniqueMapping(
        ShapeDescriptor::utilities::DuplicateMapping mapping,
        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors) {
    ShapeDescriptor::cpu::array<size_t> targetIndices = ShapeDescriptor::cpu::array<size_t>(descriptors.length);
    ShapeDescriptor::cpu::array<signed long long> duplicateIndices = ShapeDescriptor::copy::deviceArrayToHost(mapping.mappedIndices);


    size_t nextIndex = 0;
    for(size_t i = 0; i < descriptors.length; i++) {
        if(duplicateIndices.content[i] == -1) {
            // image is unique, so we give it a new place in the list.
            targetIndices.content[i] = nextIndex;
            nextIndex++;
        } else {
            // image is not unique, thus another image must have come before it.
            // We can therefore use the target index of that other image.
            targetIndices.content[i] = targetIndices.content[duplicateIndices.content[i]];
        }
    }

    ShapeDescriptor::free::array(duplicateIndices);

    ShapeDescriptor::gpu::array<size_t> device_targetIndices = ShapeDescriptor::copy::hostArrayToDevice(targetIndices);
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> targetDescriptors(mapping.uniqueElementCount);

    mapImages<<<descriptors.length, 32>>>(descriptors, targetDescriptors, device_targetIndices);

    checkCudaErrors(cudaDeviceSynchronize());

    ShapeDescriptor::free::array(targetIndices);
    ShapeDescriptor::free::array(device_targetIndices);

    return targetDescriptors;
}