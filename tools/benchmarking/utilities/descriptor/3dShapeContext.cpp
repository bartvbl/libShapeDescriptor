#include "3dShapeContext.h"
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/gpu/3dShapeContextGenerator.cuh>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/meshSampler.h>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/free/pointCloud.h>
#include <vector>
#include <map>
#include <variant>
#include <ctime>
#include <chrono>

ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> Benchmarking::utilities::descriptor::generate3DShapeContextDescriptor(
    ShapeDescriptor::cpu::Mesh mesh,
    std::string hardware,
    size_t sampleCount,
    size_t randomSeed,
    float pointDensityRadius,
    float minSupportRadius,
    float maxSupportRadius,
    std::chrono::duration<double> &elapsedTime)
{
    std::cout << "Generating 3D Shape Context Descriptor" << std::endl;

    ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> descriptor;
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOrigins = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(mesh);
    ShapeDescriptor::gpu::Mesh deviceMesh;
    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> deviceSpinOrigins;

    if (hardware == "gpu")
    {
        deviceMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);

        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> tempOrigins = ShapeDescriptor::copy::hostArrayToDevice(spinOrigins);
        deviceSpinOrigins = {tempOrigins.length, reinterpret_cast<ShapeDescriptor::OrientedPoint *>(tempOrigins.content)};

        ShapeDescriptor::gpu::PointCloud pointCloud = ShapeDescriptor::utilities::sampleMesh(deviceMesh, sampleCount, randomSeed);

        std::chrono::steady_clock::time_point descriptorTimeStart = std::chrono::steady_clock::now();
        ShapeDescriptor::gpu::array<ShapeDescriptor::ShapeContextDescriptor> descriptorGPU =
            ShapeDescriptor::gpu::generate3DSCDescriptors(pointCloud, deviceSpinOrigins, pointDensityRadius, minSupportRadius, maxSupportRadius);
        std::chrono::steady_clock::time_point descriptorTimeEnd = std::chrono::steady_clock::now();

        elapsedTime = descriptorTimeEnd - descriptorTimeStart;

        descriptor = ShapeDescriptor::copy::deviceArrayToHost(descriptorGPU);

        ShapeDescriptor::free::array(descriptorGPU);
        ShapeDescriptor::free::array(tempOrigins);
        ShapeDescriptor::free::mesh(deviceMesh);
        ShapeDescriptor::free::pointCloud(pointCloud);
    }
    else
    {
        descriptor = NULL;
    }

    ShapeDescriptor::free::array(spinOrigins);

    return descriptor;
}