#include "spinImage.h"
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/free/pointCloud.h>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/cpu/spinImageGenerator.h>
#include <shapeDescriptor/gpu/spinImageGenerator.cuh>
#include <shapeDescriptor/utilities/meshSampler.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <vector>
#include <variant>
#include <map>
#include <ctime>
#include <chrono>

ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> Benchmarking::utilities::descriptor::generateSpinImageDescriptor(
    ShapeDescriptor::cpu::Mesh mesh,
    std::string hardware,
    float supportRadius,
    float supportAngleDegrees,
    size_t sampleCount,
    size_t randomSeed,
    std::chrono::duration<double> &elapsedTime)
{
    std::cout << "Generating Spin Images Descriptor" << std::endl;

    ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> descriptor;
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOrigins = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(mesh);

    if (hardware == "gpu")
    {
        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> deviceSpinOrigins = spinOrigins.copyToGPU();

        ShapeDescriptor::gpu::Mesh deviceMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);

        ShapeDescriptor::gpu::PointCloud pointCloud = ShapeDescriptor::utilities::sampleMesh(deviceMesh, sampleCount, randomSeed);

        std::chrono::steady_clock::time_point descriptorTimeStart = std::chrono::steady_clock::now();
        ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> descriptorGPU = ShapeDescriptor::gpu::generateSpinImages(
            pointCloud, deviceSpinOrigins, supportRadius, supportAngleDegrees);
        std::chrono::steady_clock::time_point descriptorTimeEnd = std::chrono::steady_clock::now();

        elapsedTime = descriptorTimeEnd - descriptorTimeStart;

        descriptor = ShapeDescriptor::copy::deviceArrayToHost(descriptorGPU);

        ShapeDescriptor::free::array(descriptorGPU);
        ShapeDescriptor::free::array(deviceSpinOrigins);
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
