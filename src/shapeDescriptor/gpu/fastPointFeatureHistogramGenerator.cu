// FPFH implementation was partially adapted from two sources.

// Source 1: https://github.com/ahmorsi/Fast-Point-Feature-Histograms

// Source 2: Point Cloud Library

/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */


#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <helper_cuda.h>
#include <helper_math.h>
#include <cuda_runtime.h>
#endif

#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>
#include <chrono>

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
__device__ __host__ __forceinline__
bool pcl_computePairFeatures (
    const float3& histogramOriginVertex,
    const float3& histogramOriginNormal,
    const float3& samplePointVertex,
    const float3& samplePointNormal,
    float &feature1_theta,
    float &feature2_alpha,
    float &feature3_phi,
    float &feature4_euclideanDistance)
{
    feature1_theta = 0.0f;
    feature2_alpha = 0.0f;
    feature3_phi = 0.0f;
    feature4_euclideanDistance = 0.0f;

    float3 dp2p1 = samplePointVertex - histogramOriginVertex;
    feature4_euclideanDistance = length(dp2p1);

    if (feature4_euclideanDistance == 0.f) {
        return false;
    }

    float3 n1_copy = histogramOriginNormal, n2_copy = samplePointNormal;
    float angle1 = dot(n1_copy, dp2p1) / feature4_euclideanDistance;


    float angle2 = dot(n2_copy, dp2p1) / feature4_euclideanDistance;
    if (std::acos (std::abs (angle1)) > std::acos (std::abs (angle2))) {
        // switch p1 and p2
        n1_copy = samplePointNormal;
        n2_copy = histogramOriginNormal;
        dp2p1 *= (-1);
        feature3_phi = -angle2;
    } else {
        feature3_phi = angle1;
    }

    // Create a Darboux frame coordinate system u-v-w
    // u = n1; v = (p_idx - q_idx) x u / || (p_idx - q_idx) x u ||; w = u x v
    float3 v = cross(dp2p1, n1_copy);
    float v_norm = length(v);
    if (v_norm == 0.0f) {
        return false;
    }

    // Normalize v
    v *= 1.f/v_norm;

    // Do not have to normalize w - it is a unit vector by construction
    feature2_alpha = dot(v, n2_copy);

    float3 w = cross(n1_copy, v);
    // Compute f1 = arctan (w * n2, u * n2) i.e. angle of n2 in the x=u, y=w coordinate system
    feature1_theta = std::atan2 (dot(w, n2_copy), dot(n1_copy, n2_copy)); // @todo optimize this

    return true;
}

__global__ void computeSPFHHistograms(
        ShapeDescriptor::gpu::VertexList descriptorOriginVertices,
        ShapeDescriptor::gpu::VertexList descriptorOriginNormals,
        ShapeDescriptor::gpu::PointCloud pointCloud,
        const float supportRadius,
        float* histograms) {
    // Launch dimensions: one block for every descriptor
#define SPFHDescriptorIndex blockIdx.x

    __shared__ unsigned int histogramSPFH[3 * FPFH_BINS_PER_FEATURE];

    unsigned int neighbourCount = 0;

    for(int i = threadIdx.x; i < 3 * FPFH_BINS_PER_FEATURE; i+= blockDim.x) {
        histogramSPFH[i] = 0;
    }

    __syncthreads();

    float3 descriptorOrigin = descriptorOriginVertices.at(SPFHDescriptorIndex);
    float3 descriptorOriginNormal = descriptorOriginNormals.at(SPFHDescriptorIndex);

    for(unsigned int i = threadIdx.x; i < pointCloud.vertices.length; i += blockDim.x) {
        float3 pointCloudPoint = pointCloud.vertices.at(i);
        float distanceToPoint = length(pointCloudPoint - descriptorOrigin);
        if(distanceToPoint > 0 && distanceToPoint <= supportRadius) {
            float3 pointCloudNormal = pointCloud.normals.at(i);
            float feature1_theta = 0;
            float feature2_alpha = 0;
            float feature3_phi = 0;
            float feature4_euclideanDistance = 0;
            const bool featuresComputedSuccessfully = pcl_computePairFeatures(
                descriptorOrigin,
                descriptorOriginNormal,
                pointCloudPoint,
                pointCloudNormal,
                feature1_theta, feature2_alpha, feature3_phi, feature4_euclideanDistance);
            if(featuresComputedSuccessfully) {
                neighbourCount++;

                unsigned int feature1BinIndex = std::floor (FPFH_BINS_PER_FEATURE * ((feature1_theta + M_PI) * (1.0f / (2.0f * M_PI))));
                feature1BinIndex = clamp(feature1BinIndex, 0U, FPFH_BINS_PER_FEATURE - 1U);
                atomicAdd(histogramSPFH + 0 * FPFH_BINS_PER_FEATURE + feature1BinIndex, 1);

                unsigned int feature2BinIndex = std::floor (FPFH_BINS_PER_FEATURE * ((feature2_alpha + 1.0f) * 0.5f));
                feature2BinIndex = clamp(feature2BinIndex, 0U, FPFH_BINS_PER_FEATURE - 1U);
                atomicAdd(histogramSPFH + 1 * FPFH_BINS_PER_FEATURE + feature2BinIndex, 1);

                unsigned int feature3BinIndex = std::floor (FPFH_BINS_PER_FEATURE * ((feature3_phi + 1.0f) * 0.5f));
                feature3BinIndex = clamp(feature3BinIndex, 0U, FPFH_BINS_PER_FEATURE - 1U);
                atomicAdd(histogramSPFH + 2 * FPFH_BINS_PER_FEATURE + feature3BinIndex, 1);
            }
        }
    }

    __syncthreads();

    float normalisationFactor = 1.0f / float(neighbourCount);

    if(isnan(normalisationFactor)) {
        normalisationFactor = 0;
    }

    // Compute final histogram and copy it to main memory
    for(int i = threadIdx.x; i < 3 * FPFH_BINS_PER_FEATURE; i += blockDim.x) {
        histograms[SPFHDescriptorIndex * 3 * FPFH_BINS_PER_FEATURE + i] = normalisationFactor * histogramSPFH[i];
    }
}

__global__ void computeFPFHHistograms(
        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
        ShapeDescriptor::gpu::PointCloud pointCloud,
        const float supportRadius,
        float* histogramOriginHistograms,
        float* pointCloudHistograms,
        ShapeDescriptor::FPFHDescriptor* fpfhHistograms) {
    // Launch dimensions: one block for every descriptor
    // Blocks should contain just about as many threads as a histogram has bins
#define FPFHDescriptorIndex blockIdx.x

    __shared__ float histogramFPFH[3 * FPFH_BINS_PER_FEATURE];

    for(int i = threadIdx.x; i < 3 * FPFH_BINS_PER_FEATURE; i+= blockDim.x) {
        histogramFPFH[i] = histogramOriginHistograms[FPFHDescriptorIndex * 3 * FPFH_BINS_PER_FEATURE + i];
    }

    unsigned int neighbourCount = 0;

    __syncthreads();

    float3 descriptorOrigin = descriptorOrigins.content[FPFHDescriptorIndex].vertex;

    for(unsigned int neighbourHistogram = 0; neighbourHistogram < pointCloud.vertices.length; neighbourHistogram++) {
        for(unsigned int histogramBin = threadIdx.x; histogramBin < 3 * FPFH_BINS_PER_FEATURE; histogramBin += blockDim.x) {
            float3 pointCloudPoint = pointCloud.vertices.at(neighbourHistogram);
            float distanceToPoint = length(pointCloudPoint - descriptorOrigin);
            if(distanceToPoint <= supportRadius) {
                neighbourCount++;
                float distanceWeight = 1.0f / distanceToPoint;
                float histogramValue = pointCloudHistograms[neighbourHistogram * 3 * FPFH_BINS_PER_FEATURE + histogramBin];
                atomicAdd(&histogramFPFH[histogramBin], distanceWeight * histogramValue);
            }
        }
    }

    __syncthreads();

    // Copy histogram back to main memory
    for(int i = threadIdx.x; i < 3 * FPFH_BINS_PER_FEATURE; i += blockDim.x) {
        fpfhHistograms[FPFHDescriptorIndex].contents[i] = neighbourCount == 0 ? 0 : histogramFPFH[i] / float(neighbourCount);
    }
}

__global__ void reformatOrigins(
        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> originsArray,
        ShapeDescriptor::gpu::VertexList reformattedOriginVerticesList,
        ShapeDescriptor::gpu::VertexList reformattedOriginNormalsList) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index >= originsArray.length) {
        return;
    }
    ShapeDescriptor::OrientedPoint origin = originsArray.content[index];
    reformattedOriginVerticesList.set(index, origin.vertex);
    reformattedOriginNormalsList.set(index, origin.normal);
}
#endif

ShapeDescriptor::gpu::array<ShapeDescriptor::FPFHDescriptor> ShapeDescriptor::generateFPFHHistograms(
        ShapeDescriptor::gpu::PointCloud device_pointCloud,
        ShapeDescriptor::gpu::array<OrientedPoint> device_descriptorOrigins,
        float supportRadius,
        ShapeDescriptor::FPFHExecutionTimes* executionTimes)
{
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    auto totalExecutionTimeStart = std::chrono::steady_clock::now();

    size_t singleHistogramSizeBytes = sizeof(ShapeDescriptor::FPFHDescriptor);
    size_t outputHistogramsSize = device_descriptorOrigins.length * singleHistogramSizeBytes;


    // Reformat origins to a better buffer format (makes SPFH kernel usable for both input buffers)
    std::cout << "\t\t\tReformatting origins.." << std::endl;
    auto reformatTimeStart = std::chrono::steady_clock::now();
    gpu::VertexList device_reformattedOriginVerticesList(device_descriptorOrigins.length);
    gpu::VertexList device_reformattedOriginNormalsList(device_descriptorOrigins.length);
    reformatOrigins<<<(device_descriptorOrigins.length / 32) + 1, 32>>>(
            device_descriptorOrigins,
            device_reformattedOriginVerticesList,
            device_reformattedOriginNormalsList);
    checkCudaErrors(cudaDeviceSynchronize());

    std::chrono::milliseconds originReformatDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - reformatTimeStart);

    // Compute SPFH for descriptor origin points
    std::cout << "\t\t\tGenerating SPFH histograms for origin vertices.." << std::endl;
    auto originsSPFHTimeStart = std::chrono::steady_clock::now();
    float* device_origins_SPFH_histograms;
    checkCudaErrors(cudaMalloc(&device_origins_SPFH_histograms, outputHistogramsSize));
    computeSPFHHistograms<<<device_descriptorOrigins.length, 64>>>(
            device_reformattedOriginVerticesList,
            device_reformattedOriginNormalsList,
            device_pointCloud,
            supportRadius,
            device_origins_SPFH_histograms);
    checkCudaErrors(cudaDeviceSynchronize());

    std::chrono::milliseconds originsSPFHDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - originsSPFHTimeStart);

    device_reformattedOriginVerticesList.free();
    device_reformattedOriginNormalsList.free();

    // Compute SPFH for all points in point cloud
    std::cout << "\t\t\tGenerating SPFH histograms for point cloud vertices.." << std::endl;
    auto pointCloudSPFHTimeStart = std::chrono::steady_clock::now();
    float* device_pointCloud_SPFH_histograms;
    checkCudaErrors(cudaMalloc(&device_pointCloud_SPFH_histograms, device_pointCloud.pointCount * singleHistogramSizeBytes));
    computeSPFHHistograms<<<device_pointCloud.pointCount, 64>>>(
            device_pointCloud.vertices,
            device_pointCloud.normals,
            device_pointCloud,
            supportRadius,
            device_pointCloud_SPFH_histograms);
    checkCudaErrors(cudaDeviceSynchronize());

    std::chrono::milliseconds pointCloudSPFHDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - pointCloudSPFHTimeStart);

    // Compute FPFH
    std::cout << "\t\t\tGenerating FPFH descriptors.." << std::endl;
    auto fpfhGenerationTimeStart = std::chrono::steady_clock::now();
    ShapeDescriptor::gpu::array<ShapeDescriptor::FPFHDescriptor> device_histograms;
    device_histograms.length = device_descriptorOrigins.length;
    checkCudaErrors(cudaMalloc(&device_histograms.content, outputHistogramsSize));

    computeFPFHHistograms<<<device_descriptorOrigins.length, 64>>>(
            device_descriptorOrigins,
            device_pointCloud,
            supportRadius,
            device_origins_SPFH_histograms,
            device_pointCloud_SPFH_histograms,
            device_histograms.content);
    checkCudaErrors(cudaDeviceSynchronize());

    std::chrono::milliseconds fpfhHistogramComputationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - fpfhGenerationTimeStart);

    checkCudaErrors(cudaFree(device_origins_SPFH_histograms));
    checkCudaErrors(cudaFree(device_pointCloud_SPFH_histograms));

    std::chrono::milliseconds totalExecutionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - totalExecutionTimeStart);

    if(executionTimes != nullptr) {
        executionTimes->originReformatExecutionTimeSeconds = double(originReformatDuration.count()) / 1000.0;
        executionTimes->originSPFHGenerationExecutionTimeSeconds = double(originsSPFHDuration.count()) / 1000.0;
        executionTimes->pointCloudSPFHGenerationExecutionTimeSeconds = double(pointCloudSPFHDuration.count()) / 1000.0;
        executionTimes->fpfhGenerationExecutionTimeSeconds = double(fpfhHistogramComputationDuration.count()) / 1000.0;
        executionTimes->totalExecutionTimeSeconds = double(totalExecutionDuration.count()) / 1000.0;
    }

    return device_histograms;
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}