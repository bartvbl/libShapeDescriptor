#include "duplicateRemoval.cuh"

#include <vector>
#include <nvidia/helper_cuda.h>
#include <cassert>
#include <spinImage/utilities/copy/mesh.h>

__host__ __device__ __inline__ size_t roundSizeToNearestCacheLine(size_t sizeInBytes) {
    return (sizeInBytes + 127u) & ~((size_t) 127);
}


__global__ void detectDuplicates(SpinImage::gpu::Mesh mesh, bool* isDuplicate) {
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

__global__ void computeTargetIndices(SpinImage::array<signed long long> targetIndices, bool* duplicateVertices, size_t vertexCount) {
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

SpinImage::array<signed long long> SpinImage::utilities::computeUniqueIndexMapping(SpinImage::gpu::Mesh boxScene, std::vector<SpinImage::gpu::Mesh> deviceMeshes, std::vector<size_t> *uniqueVertexCounts, size_t &totalUniqueVertexCount) {
    size_t sceneVertexCount = boxScene.vertexCount;

    bool* device_duplicateVertices;
    checkCudaErrors(cudaMalloc(&device_duplicateVertices, sceneVertexCount * sizeof(bool)));
    detectDuplicates<<<(boxScene.vertexCount / 256) + 1, 256>>>(boxScene, device_duplicateVertices);
    checkCudaErrors(cudaDeviceSynchronize());

    bool* temp_duplicateVertices = new bool[sceneVertexCount];
    checkCudaErrors(cudaMemcpy(temp_duplicateVertices, device_duplicateVertices, boxScene.vertexCount * sizeof(bool), cudaMemcpyDeviceToHost));

    //std::fstream tempOutFile("DEBUG_duplicates_" + std::to_string(boxScene.vertexCount) + ".txt", std::ios::out);

    SpinImage::cpu::Mesh temp_host_boxScene = SpinImage::copy::deviceMeshToHost(boxScene);

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
            SpinImage::cpu::float3 vertex = temp_host_boxScene.vertices[baseIndex + i];
            SpinImage::cpu::float3 normal = temp_host_boxScene.normals[baseIndex + i];

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

    SpinImage::cpu::freeMesh(temp_host_boxScene);
    delete[] temp_duplicateVertices;

    SpinImage::array<signed long long> device_uniqueIndexMapping;
    device_uniqueIndexMapping.length = boxScene.vertexCount;
    checkCudaErrors(cudaMalloc(&device_uniqueIndexMapping.content, boxScene.vertexCount * sizeof(signed long long)));
    computeTargetIndices<<<(boxScene.vertexCount / 256) + 1, 256>>>(device_uniqueIndexMapping, device_duplicateVertices, boxScene.vertexCount);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(device_duplicateVertices));

    return device_uniqueIndexMapping;
}

__global__ void mapVertices(SpinImage::gpu::Mesh boxScene, SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> origins, SpinImage::array<signed long long> mapping) {
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

        SpinImage::gpu::DeviceOrientedPoint origin;
        origin.vertex = vertex;
        origin.normal = normal;

        origins.content[targetIndex] = origin;
    }
}

SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> SpinImage::utilities::applyUniqueMapping(SpinImage::gpu::Mesh boxScene, SpinImage::array<signed long long> device_mapping, size_t totalUniqueVertexCount) {
    assert(boxScene.vertexCount == device_mapping.length);

    SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> device_origins;
    device_origins.length = totalUniqueVertexCount;
    checkCudaErrors(cudaMalloc(&device_origins.content, totalUniqueVertexCount * sizeof(SpinImage::gpu::DeviceOrientedPoint)));

    mapVertices<<<(boxScene.vertexCount / 256) + 1, 256>>>(boxScene, device_origins, device_mapping);
    checkCudaErrors(cudaDeviceSynchronize());

    return device_origins;
}

SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> SpinImage::utilities::computeUniqueVertices(SpinImage::gpu::Mesh &mesh) {
    std::vector<SpinImage::gpu::Mesh> deviceMeshes;
    deviceMeshes.push_back(mesh);
    std::vector<size_t> vertexCounts;
    size_t totalUniqueVertexCount;
    SpinImage::array<signed long long> device_mapping = computeUniqueIndexMapping(mesh, deviceMeshes, &vertexCounts, totalUniqueVertexCount);
    SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> device_origins = applyUniqueMapping(mesh, device_mapping, totalUniqueVertexCount);
    checkCudaErrors(cudaFree(device_mapping.content));
    return device_origins;
}