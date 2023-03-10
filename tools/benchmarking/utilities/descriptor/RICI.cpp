#include "RICI.h"
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <vector>
#include <variant>
#include <ctime>
#include <chrono>

ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> Benchmarking::utilities::descriptor::generateRICIDescriptor(
    ShapeDescriptor::cpu::Mesh mesh,
    std::string hardware,
    float supportRadius,
    std::chrono::duration<double> &elapsedTime)
{
    std::cout << "Generating Radial Intersection Count Images Descriptor" << std::endl;

    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptor;
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOrigins = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(mesh);
    ShapeDescriptor::gpu::Mesh deviceMesh;
    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> deviceSpinOrigins;

    if (hardware == "gpu")
    {
        deviceMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);

        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> tempOrigins = ShapeDescriptor::copy::hostArrayToDevice(spinOrigins);
        deviceSpinOrigins = {tempOrigins.length, reinterpret_cast<ShapeDescriptor::OrientedPoint *>(tempOrigins.content)};

        std::chrono::steady_clock::time_point descriptorTimeStart = std::chrono::steady_clock::now();
        ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> descriptorGPU =
            ShapeDescriptor::gpu::generateRadialIntersectionCountImages(deviceMesh, deviceSpinOrigins, supportRadius);
        std::chrono::steady_clock::time_point descriptorTimeEnd = std::chrono::steady_clock::now();

        elapsedTime = descriptorTimeEnd - descriptorTimeStart;

        descriptor = ShapeDescriptor::copy::deviceArrayToHost(descriptorGPU);

        ShapeDescriptor::free::array(descriptorGPU);
        ShapeDescriptor::free::array(tempOrigins);
        ShapeDescriptor::free::mesh(deviceMesh);
    }
    else
    {
        std::chrono::steady_clock::time_point descriptorTimeStart = std::chrono::steady_clock::now();
        descriptor = ShapeDescriptor::cpu::generateRadialIntersectionCountImages(mesh, spinOrigins, supportRadius);
        std::chrono::steady_clock::time_point descriptorTimeEnd = std::chrono::steady_clock::now();

        elapsedTime = descriptorTimeEnd - descriptorTimeStart;
    }

    ShapeDescriptor::free::array(spinOrigins);

    return descriptor;
}
