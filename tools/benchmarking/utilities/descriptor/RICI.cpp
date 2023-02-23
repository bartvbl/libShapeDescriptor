#include "RICI.h"
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <vector>
#include <variant>

std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>> Benchmarking::utilities::descriptor::generateRICIDescriptors(ShapeDescriptor::cpu::Mesh meshOne, ShapeDescriptor::cpu::Mesh meshTwo, std::vector<std::variant<int, std::string>> metadata, std::string hardware)
{
    std::cout << "Generating Radial Intersection Count Images Descriptors" << std::endl;
    float supportRadius = 0.3f;

    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOriginsOne = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(meshOne);
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOriginsTwo = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(meshTwo);

    ShapeDescriptor::gpu::Mesh deviceMeshOne;
    ShapeDescriptor::gpu::Mesh deviceMeshTwo;
    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> deviceSpinOriginsOne;
    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> deviceSpinOriginsTwo;

    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptorsOne;
    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptorsTwo;

    if (hardware == "gpu")
    {
        deviceMeshOne = ShapeDescriptor::copy::hostMeshToDevice(meshOne);
        deviceMeshTwo = ShapeDescriptor::copy::hostMeshToDevice(meshTwo);

        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> tempOriginsOne = ShapeDescriptor::copy::hostArrayToDevice(spinOriginsOne);
        deviceSpinOriginsOne = {tempOriginsOne.length, reinterpret_cast<ShapeDescriptor::OrientedPoint *>(tempOriginsOne.content)};

        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> tempOriginsTwo = ShapeDescriptor::copy::hostArrayToDevice(spinOriginsTwo);
        deviceSpinOriginsTwo = {tempOriginsTwo.length, reinterpret_cast<ShapeDescriptor::OrientedPoint *>(tempOriginsTwo.content)};

        ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> dOne =
            ShapeDescriptor::gpu::generateRadialIntersectionCountImages(deviceMeshOne, deviceSpinOriginsOne, supportRadius);
        descriptorsOne = ShapeDescriptor::copy::deviceArrayToHost(dOne);

        ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> dTwo =
            ShapeDescriptor::gpu::generateRadialIntersectionCountImages(deviceMeshTwo, deviceSpinOriginsTwo, supportRadius);
        descriptorsTwo = ShapeDescriptor::copy::deviceArrayToHost(dTwo);

        ShapeDescriptor::free::array(dOne);
        ShapeDescriptor::free::array(dTwo);
    }
    else
    {
        descriptorsOne = ShapeDescriptor::cpu::generateRadialIntersectionCountImages(meshOne, spinOriginsOne, supportRadius);
        descriptorsTwo = ShapeDescriptor::cpu::generateRadialIntersectionCountImages(meshTwo, spinOriginsTwo, supportRadius);
    }

    std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>> descriptors;
    descriptors[0] = descriptorsOne;
    descriptors[1] = descriptorsTwo;

    ShapeDescriptor::free::array(spinOriginsOne);
    ShapeDescriptor::free::array(spinOriginsTwo);

    return descriptors;
}
