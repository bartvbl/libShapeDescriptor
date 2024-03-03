#include <catch2/catch_test_macros.hpp>
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <cuda_runtime.h>
#include <vector_types.h>

#include <shapeDescriptor/shapeDescriptor.h>
#include <iostream>


unsigned int *computeNeighbourCounts(const float *simpleCloud, unsigned int pointCount, float radius) {
    ShapeDescriptor::gpu::PointCloud device_pointCloud(pointCount);

    cudaMemcpy(device_pointCloud.vertices.array, simpleCloud, pointCount * 3 * sizeof(float), cudaMemcpyHostToDevice);

    ShapeDescriptor::gpu::array<unsigned int> device_pointDensities = ShapeDescriptor::computePointDensities(radius, device_pointCloud);

    unsigned int* counts = new unsigned int[pointCount];
    cudaMemcpy(counts, device_pointDensities.content, pointCount * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    return counts;
}

TEST_CASE("Counting the number of points in the vicinity of others")
{
    SECTION("Simple point cloud") {
        // note: VERTICAL definitions!
        std::unique_ptr<float[]> simpleCloud {new float[12]{
            -1,  1,  1, -1,
            -1, -1,  1,  1,
             0,  0,  0,  0
        }};

        std::unique_ptr<unsigned int[]> counts {computeNeighbourCounts(simpleCloud.get(), 4, 5)};

        for(int i = 0; i < 4; i++) {
            REQUIRE(counts.get()[i] == 3);
        }

    }

    SECTION("Simple point cloud, small radius") {
        // note: VERTICAL definitions!
        std::unique_ptr<float[]> simpleCloud {new float[12]{
                -1,  1,  1, -1,
                -1, -1,  1,  1,
                 0,  0,  0,  0
        }};

        std::unique_ptr<unsigned int[]> counts {computeNeighbourCounts(simpleCloud.get(), 4, 2)};

        for(int i = 0; i < 4; i++) {
            REQUIRE(counts.get()[i] == 2);
        }
    }

    SECTION("Simple point cloud, even smaller radius") {
        // note: VERTICAL definitions!
        std::unique_ptr<float[]> simpleCloud {new float[12]{
                -1,  1,  1, -1,
                -1, -1,  1,  1,
                 0,  0,  0,  0
        }};

        std::unique_ptr<unsigned int[]> counts {computeNeighbourCounts(simpleCloud.get(), 4, 1)};

        for(int i = 0; i < 4; i++) {
            REQUIRE(counts.get()[i] == 0);
        }
    }

    SECTION("Point cloud with only 1 point") {
        // note: VERTICAL definitions!
        std::unique_ptr<float[]> simpleCloud { new float[3]{
                -1,
                -1,
                 0
        }};

        std::unique_ptr<unsigned int[]> counts {computeNeighbourCounts(simpleCloud.get(), 1, 1)};

        REQUIRE(counts.get()[0] == 0);
    }

    SECTION("Two points at the same location") {
        // note: VERTICAL definitions!
        std::unique_ptr<float[]> simpleCloud {new float[6]{
                -1, -1,
                -1, -1,
                 0,  0
        }};

        std::unique_ptr<unsigned int[]> counts {computeNeighbourCounts(simpleCloud.get(), 2, 1)};

        REQUIRE(counts.get()[0] == 1);
        REQUIRE(counts.get()[1] == 1);
    }

    SECTION("Cube point cloud, small radius") {
        // note: VERTICAL definitions!
        std::unique_ptr<float[]> simpleCloud { new float[24]{
                -1,  1,  1, -1, -1,  1,  1, -1,
                -1, -1,  1,  1, -1, -1,  1,  1,
                -1, -1, -1, -1,  1,  1,  1,  1
        }};

        std::unique_ptr<unsigned int[]> counts {computeNeighbourCounts(simpleCloud.get(), 8, 1)};

        for(int i = 0; i < 8; i++) {
            REQUIRE(counts.get()[i] == 0);
        }
    }

    SECTION("Cube point cloud, medium radius") {
        // note: VERTICAL definitions!
        std::unique_ptr<float[]> simpleCloud {new float[24]{
                -1,  1,  1, -1, -1,  1,  1, -1,
                -1, -1,  1,  1, -1, -1,  1,  1,
                -1, -1, -1, -1,  1,  1,  1,  1
        }};

        std::unique_ptr<unsigned int[]> counts { computeNeighbourCounts(simpleCloud.get(), 8, 2)};

        for(int i = 0; i < 8; i++) {
            REQUIRE(counts.get()[i] == 3);
        }
    }

    SECTION("Cube point cloud, large radius") {
        // note: VERTICAL definitions!
        std::unique_ptr<float[]> simpleCloud { new float[24]{
                -1,  1,  1, -1, -1,  1,  1, -1,
                -1, -1,  1,  1, -1, -1,  1,  1,
                -1, -1, -1, -1,  1,  1,  1,  1
        }};

        std::unique_ptr<unsigned int[]> counts { computeNeighbourCounts(simpleCloud.get(), 8, 5)};

        for(int i = 0; i < 8; i++) {
            REQUIRE(counts.get()[i] == 7);
        }
    }

    SECTION("Uneven distribution") {
        // note: VERTICAL definitions!
        std::unique_ptr<float[]> simpleCloud { new float[24]{
                -5, -4, -2, -1,  0,  3,  4,  8,
                 1,  1,  1,  1,  1,  1,  1,  1,
                 0,  0,  0,  0,  0,  0,  0,  0
        }};

        std::unique_ptr<unsigned int[]> counts { computeNeighbourCounts(simpleCloud.get(), 8, 1)};

        unsigned int correctCounts[] = {1, 1, 1, 2, 1, 1, 1, 0};

        for(int i = 0; i < 8; i++) {
            REQUIRE(counts.get()[i] == correctCounts[i]);
        }
    }
}
#endif