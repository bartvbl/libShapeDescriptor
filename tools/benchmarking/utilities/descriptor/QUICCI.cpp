#include "QUICCI.h"
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/cpu/quickIntersectionCountImageGenerator.h>
#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>
#include <benchmarking/utilities/distance/cosine.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <vector>
#include <variant>
#include <ctime>
#include <chrono>

ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> Benchmarking::utilities::descriptor::generateQUICCIDescriptor(
    ShapeDescriptor::cpu::Mesh mesh,
    std::string hardware,
    float supportRadius,
    std::chrono::duration<double> &elapsedTime)
{
    std::cout << "Generating Quick Intersection Count Change Images Descriptor" << std::endl;

    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptor;
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOrigins = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(mesh);
    ShapeDescriptor::gpu::Mesh deviceMesh;
    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> deviceSpinOrigins;

    if (hardware == "gpu")
    {
        deviceMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);

        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> tempOrigins = ShapeDescriptor::copy::hostArrayToDevice(spinOrigins);
        deviceSpinOrigins = {tempOrigins.length, reinterpret_cast<ShapeDescriptor::OrientedPoint *>(tempOrigins.content)};

        std::chrono::steady_clock::time_point descriptorTimeStart = std::chrono::steady_clock::now();
        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptorGPU =
            ShapeDescriptor::gpu::generateQUICCImages(deviceMesh, deviceSpinOrigins, supportRadius);
        std::chrono::steady_clock::time_point descriptorTimeEnd = std::chrono::steady_clock::now();

        elapsedTime = descriptorTimeEnd - descriptorTimeStart;

        descriptor = ShapeDescriptor::copy::deviceArrayToHost(descriptorGPU);

        ShapeDescriptor::free::array(descriptorGPU);
        ShapeDescriptor::free::array(tempOrigins);
        ShapeDescriptor::free::mesh(deviceMesh);
    }
    else
    {
        descriptor = ShapeDescriptor::cpu::generateQUICCImages(mesh, spinOrigins, supportRadius);
    }

    ShapeDescriptor::free::array(spinOrigins);

    return descriptor;
}