#include <catch2/catch.hpp>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <spinImage/gpu/types/PointCloud.h>
#include <spinImage/common/types/array.h>
#include <spinImage/utilities/pointCloudUtils.h>
#include <iostream>


unsigned int *computeNeighbourCounts(const float *simpleCloud, unsigned int pointCount, float radius) {
    SpinImage::gpu::PointCloud device_pointCloud(pointCount);

    cudaMemcpy(device_pointCloud.vertices.array, simpleCloud, pointCount * 3 * sizeof(float), cudaMemcpyHostToDevice);

    SpinImage::array<unsigned int> device_pointDensities = SpinImage::utilities::computePointDensities(radius, device_pointCloud);

    unsigned int* counts = new unsigned int[pointCount];
    cudaMemcpy(counts, device_pointDensities.content, pointCount * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    return counts;
}

TEST_CASE("Counting the number of points in the vicinity of others")
{
    SECTION("Simple point cloud") {
        // note: VERTICAL definitions!
        const float* simpleCloud = new float[12]{
            -1,  1,  1, -1,
            -1, -1,  1,  1,
             0,  0,  0,  0
        };

        unsigned int *counts = computeNeighbourCounts(simpleCloud, 4, 5);

        for(int i = 0; i < 4; i++) {
            REQUIRE(counts[i] == 3);
        }
    }

    SECTION("Simple point cloud, small radius") {
        // note: VERTICAL definitions!
        const float* simpleCloud = new float[12]{
                -1,  1,  1, -1,
                -1, -1,  1,  1,
                0,  0,  0,  0
        };

        unsigned int *counts = computeNeighbourCounts(simpleCloud, 4, 2);

        for(int i = 0; i < 4; i++) {
            REQUIRE(counts[i] == 2);
        }
    }

    SECTION("Simple point cloud, even smaller radius") {
        // note: VERTICAL definitions!
        const float* simpleCloud = new float[12]{
                -1,  1,  1, -1,
                -1, -1,  1,  1,
                0,  0,  0,  0
        };

        unsigned int *counts = computeNeighbourCounts(simpleCloud, 4, 1);

        for(int i = 0; i < 4; i++) {
            REQUIRE(counts[i] == 0);
        }
    }
}