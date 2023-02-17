#include "QUICCI.h"
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/cpu/quickIntersectionCountImageGenerator.h>
#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>
#include <benchmarking/utilities/distance/cosine.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <vector>
#include <variant>

std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor>> Benchmarking::utilities::descriptor::generateQUICCIDescriptors(ShapeDescriptor::cpu::Mesh meshOne, ShapeDescriptor::cpu::Mesh meshTwo, std::vector<std::variant<int, std::string>> metadata, std::string hardware)
{
    float supportRadius = 0.3f;
    std::cout << "Generating Quick Intersection Count Change Images" << std::endl;

    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOriginsOne = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(meshOne);
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOriginsTwo = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(meshTwo);

    ShapeDescriptor::gpu::Mesh deviceMeshOne;
    ShapeDescriptor::gpu::Mesh deviceMeshTwo;
    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> deviceSpinOriginsOne;
    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> deviceSpinOriginsTwo;

    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptorsOne;
    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptorsTwo;

    if (hardware == "gpu")
    {
        deviceMeshOne = ShapeDescriptor::copy::hostMeshToDevice(meshOne);
        deviceMeshTwo = ShapeDescriptor::copy::hostMeshToDevice(meshTwo);

        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> tempOriginsOne = ShapeDescriptor::copy::hostArrayToDevice(spinOriginsOne);
        deviceSpinOriginsOne = {tempOriginsOne.length, reinterpret_cast<ShapeDescriptor::OrientedPoint *>(tempOriginsOne.content)};

        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> tempOriginsTwo = ShapeDescriptor::copy::hostArrayToDevice(spinOriginsTwo);
        deviceSpinOriginsTwo = {tempOriginsTwo.length, reinterpret_cast<ShapeDescriptor::OrientedPoint *>(tempOriginsTwo.content)};

        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> dOne =
            ShapeDescriptor::gpu::generateQUICCImages(deviceMeshOne, deviceSpinOriginsOne, supportRadius);

        descriptorsOne = ShapeDescriptor::copy::deviceArrayToHost(dOne);

        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> dTwo =
            ShapeDescriptor::gpu::generateQUICCImages(deviceMeshTwo, deviceSpinOriginsTwo, supportRadius);

        descriptorsTwo = ShapeDescriptor::copy::deviceArrayToHost(dTwo);

        ShapeDescriptor::free::array(dOne);
        ShapeDescriptor::free::array(dTwo);
    }
    else
    {
        descriptorsOne = ShapeDescriptor::cpu::generateQUICCImages(meshOne, spinOriginsOne, 0.3f);
        descriptorsTwo = ShapeDescriptor::cpu::generateQUICCImages(meshOne, spinOriginsOne, 0.3f);
    }

    std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor>> descriptors;
    descriptors[0] = descriptorsOne;
    descriptors[1] = descriptorsTwo;

    ShapeDescriptor::free::array(spinOriginsOne);
    ShapeDescriptor::free::array(spinOriginsTwo);

    return descriptors;
}