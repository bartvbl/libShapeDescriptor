#include "spinImage.h"
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/free/pointCloud.h>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/cpu/spinImageGenerator.h>
#include <shapeDescriptor/gpu/spinImageGenerator.cuh>
#include <shapeDescriptor/utilities/meshSampler.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <vector>
#include <variant>
#include <map>

std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor>> Benchmarking::utilities::descriptor::generateSpinImageDescriptors(ShapeDescriptor::cpu::Mesh meshOne, ShapeDescriptor::cpu::Mesh meshTwo, std::vector<std::variant<int, std::string>> metadata, std::string hardware)
{
    std::cout << "Generating Spin Images Descriptors" << std::endl;
    size_t sampleCount = 1000000;
    size_t randomSeed = 5553580318008;
    float supportRadius = 0.3f;
    float supportAngleDegrees = 10.0f;

    ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> descriptorsOne;
    ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> descriptorsTwo;

    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOriginsOne = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(meshOne);
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOriginsTwo = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(meshTwo);

    if (hardware == "gpu")
    {
        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> tempOriginsOne = ShapeDescriptor::copy::hostArrayToDevice(spinOriginsOne);
        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> deviceSpinOriginsOne = {tempOriginsOne.length, reinterpret_cast<ShapeDescriptor::OrientedPoint *>(tempOriginsOne.content)};

        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> tempOriginsTwo = ShapeDescriptor::copy::hostArrayToDevice(spinOriginsTwo);
        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> deviceSpinOriginsTwo = {tempOriginsTwo.length, reinterpret_cast<ShapeDescriptor::OrientedPoint *>(tempOriginsTwo.content)};

        ShapeDescriptor::gpu::Mesh deviceMeshOne = ShapeDescriptor::copy::hostMeshToDevice(meshOne);
        ShapeDescriptor::gpu::Mesh deviceMeshTwo = ShapeDescriptor::copy::hostMeshToDevice(meshTwo);

        ShapeDescriptor::gpu::PointCloud pointCloudOne = ShapeDescriptor::utilities::sampleMesh(deviceMeshOne, sampleCount, randomSeed);
        ShapeDescriptor::gpu::PointCloud pointCloudTwo = ShapeDescriptor::utilities::sampleMesh(deviceMeshTwo, sampleCount, randomSeed);

        ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> dOne = ShapeDescriptor::gpu::generateSpinImages(pointCloudOne, deviceSpinOriginsOne, supportRadius, supportAngleDegrees);
        descriptorsOne = ShapeDescriptor::copy::deviceArrayToHost(dOne);

        ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> dTwo = ShapeDescriptor::gpu::generateSpinImages(pointCloudTwo, deviceSpinOriginsTwo, supportRadius, supportAngleDegrees);
        descriptorsTwo = ShapeDescriptor::copy::deviceArrayToHost(dOne);

        ShapeDescriptor::free::array(dOne);
        ShapeDescriptor::free::array(dTwo);
        ShapeDescriptor::free::pointCloud(pointCloudOne);
        ShapeDescriptor::free::pointCloud(pointCloudTwo);
    }
    else
    {
        descriptorsOne = NULL;
        descriptorsTwo = NULL;
    }

    std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor>> descriptors;
    descriptors[0] = descriptorsOne;
    descriptors[1] = descriptorsTwo;

    ShapeDescriptor::free::array(spinOriginsOne);
    ShapeDescriptor::free::array(spinOriginsTwo);

    return descriptors;
}
