#include <catch2/catch.hpp>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <spinImage/gpu/types/PointCloud.h>
#include <spinImage/common/types/array.h>
#include <spinImage/utilities/pointCloudUtils.h>
#include <iostream>

TEST_CASE("Counting the number of points in the vicinity of others") {
    SECTION("Simple point cloud") {
        // note: VERTICAL definitions!
        const float* simpleCloud = new float[12]{
            -1,  1,  1, -1,
            -1, -1,  1,  1,
             0,  0,  0,  0
        };

        SpinImage::gpu::PointCloud device_pointCloud(4);

        cudaMemcpy(device_pointCloud.vertices.array, simpleCloud, 12 * sizeof(float), cudaMemcpyHostToDevice);

        SpinImage::array<unsigned int> device_pointDensities = SpinImage::utilities::computePointDensities(5, device_pointCloud);

        unsigned int* counts = new unsigned int[4];
        cudaMemcpy(counts, device_pointDensities.content, 4 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        std::cout << std::endl << "Final counts: " << counts[0] << ", " << counts[1] << ", " << counts[2] << ", " << counts[3] << std::endl << std::endl;

        for(int i = 0; i < 4; i++) {
            REQUIRE(counts[i] == 3);
        }

    }
}