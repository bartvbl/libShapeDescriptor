#include <catch2/catch_test_macros.hpp>
#include <shapeDescriptor/shapeDescriptor.h>
#include <random>
#include <shapeDescriptor/descriptors/ShapeContextGenerator.h>

TEST_CASE("Ensuring volume computation makes sense") {
    SECTION("Volume computation") {
        const float maxSupportRadius = 5;
        const float minSupportRadius = 0.1;

        float totalVolume = 0;
        for(short layer = 0; layer < SHAPE_CONTEXT_LAYER_COUNT; layer++) {
            for(short slice = 0; slice < SHAPE_CONTEXT_VERTICAL_SLICE_COUNT; slice++) {
                totalVolume += ShapeDescriptor::internal::computeBinVolume(slice, layer, minSupportRadius, maxSupportRadius);
            }
        }
        totalVolume *= float(SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT);

        const float largeSphereVolume = (4.0f / 3.0f) * float(M_PI) * maxSupportRadius * maxSupportRadius * maxSupportRadius;
        const float smallSphereVolume = (4.0f / 3.0f) * float(M_PI) * minSupportRadius * minSupportRadius * minSupportRadius;
        const float supportVolume = largeSphereVolume - smallSphereVolume;

        float volumeDelta = totalVolume - supportVolume;

        REQUIRE(volumeDelta < 0.0001);
    }
}

TEST_CASE("3D Shape Context") {
    SECTION("CPU equivalent to GPU") {
        const int vertexCount = 150;
        const int descriptorCount = 100;
        const int randomSeed = 123456;

        ShapeDescriptor::cpu::Mesh randomMesh(vertexCount);
        static_assert(vertexCount % 3 == 0);
        std::mt19937_64 randomEngine(randomSeed);
        std::uniform_real_distribution<float> distribution(-5, 5);
        for(uint32_t i = 0; i < randomMesh.vertexCount; i+=3) {
            ShapeDescriptor::cpu::float3 vertex0 = {distribution(randomEngine), distribution(randomEngine), distribution(randomEngine)};
            ShapeDescriptor::cpu::float3 vertex1 = {distribution(randomEngine), distribution(randomEngine), distribution(randomEngine)};
            ShapeDescriptor::cpu::float3 vertex2 = {distribution(randomEngine), distribution(randomEngine), distribution(randomEngine)};
            ShapeDescriptor::cpu::float3 normal = ShapeDescriptor::computeTriangleNormal(vertex0, vertex1, vertex2);
            randomMesh.vertices[i + 0] = vertex0;
            randomMesh.vertices[i + 1] = vertex1;
            randomMesh.vertices[i + 2] = vertex2;
            randomMesh.normals[i + 0] = normal;
            randomMesh.normals[i + 1] = normal;
            randomMesh.normals[i + 2] = normal;
        }

        ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> origins(descriptorCount);
        for(uint32_t i = 0; i < descriptorCount; i++) {
            ShapeDescriptor::OrientedPoint point;
            point.vertex = {distribution(randomEngine), distribution(randomEngine), distribution(randomEngine)};
            point.normal = {distribution(randomEngine), distribution(randomEngine), distribution(randomEngine)};
            point.normal = normalize(point.normal);
            origins[i] = point;
        }

        ShapeDescriptor::cpu::PointCloud cpuCloud = ShapeDescriptor::sampleMesh(randomMesh, 1000000, randomSeed);

        const float pointDensityRadius = 0.048;
        const float minSupportRadius = (0.1 / 2.5) * 0.096;
        const float maxSupportRadius = 0.096;

        ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> cpuDescriptors = ShapeDescriptor::generate3DSCDescriptors(cpuCloud, origins, pointDensityRadius, minSupportRadius, maxSupportRadius);

        ShapeDescriptor::gpu::PointCloud gpuCloud = ShapeDescriptor::copyToGPU(cpuCloud);
        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> gpuOrigins = ShapeDescriptor::copyToGPU(origins);

        ShapeDescriptor::gpu::array<ShapeDescriptor::ShapeContextDescriptor> gpuDescriptors = ShapeDescriptor::generate3DSCDescriptors(gpuCloud, gpuOrigins, pointDensityRadius, minSupportRadius, maxSupportRadius);
        ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> otherDescriptors = ShapeDescriptor::copyToCPU(gpuDescriptors);

        const float tolerance = 0.001;
        for(uint32_t descriptorIndex = 0; descriptorIndex < descriptorCount; descriptorIndex++) {
            for(uint32_t i = 0; i < SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT * SHAPE_CONTEXT_LAYER_COUNT * SHAPE_CONTEXT_VERTICAL_SLICE_COUNT; i++) {
                float delta = std::abs(cpuDescriptors[descriptorIndex].contents[i] - otherDescriptors[descriptorIndex].contents[i]);
                REQUIRE(delta < tolerance);
            }
        }

        ShapeDescriptor::free(cpuDescriptors);
        ShapeDescriptor::free(gpuDescriptors);
        ShapeDescriptor::free(otherDescriptors);
        ShapeDescriptor::free(randomMesh);
        ShapeDescriptor::free(cpuCloud);
        ShapeDescriptor::free(gpuCloud);
        ShapeDescriptor::free(origins);
        ShapeDescriptor::free(gpuOrigins);
        REQUIRE(true);
    }
}