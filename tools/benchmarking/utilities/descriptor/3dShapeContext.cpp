#include "3dShapeContext.h"
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/gpu/3dShapeContextGenerator.cuh>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/meshSampler.h>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/free/pointCloud.h>
#include <vector>
#include <map>
#include <variant>

std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor>> Benchmarking::utilities::descriptor::generate3DShapeContextDescriptors(ShapeDescriptor::cpu::Mesh meshOne, ShapeDescriptor::cpu::Mesh meshTwo, std::vector<std::variant<int, std::string>> metadata, std::string hardware)
{
    std::cout << "Generating 3D Shape Context Descriptors" << std::endl;
    size_t sampleCount = 1000000;
    size_t randomSeed = 5553580318008;
    float pointDensityRadius = 0.4f;
    float minSupportRadius = 0.2f;
    float maxSupportRadius = 1;

    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOriginsOne = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(meshOne);
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOriginsTwo = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(meshTwo);

    ShapeDescriptor::gpu::Mesh deviceMeshOne;
    ShapeDescriptor::gpu::Mesh deviceMeshTwo;
    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> deviceSpinOriginsOne;
    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> deviceSpinOriginsTwo;

    ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> descriptorsOne;
    ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> descriptorsTwo;

    if (hardware == "gpu")
    {
        deviceMeshOne = ShapeDescriptor::copy::hostMeshToDevice(meshOne);
        deviceMeshTwo = ShapeDescriptor::copy::hostMeshToDevice(meshTwo);

        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> tempOriginsOne = ShapeDescriptor::copy::hostArrayToDevice(spinOriginsOne);
        deviceSpinOriginsOne = {tempOriginsOne.length, reinterpret_cast<ShapeDescriptor::OrientedPoint *>(tempOriginsOne.content)};

        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> tempOriginsTwo = ShapeDescriptor::copy::hostArrayToDevice(spinOriginsTwo);
        deviceSpinOriginsTwo = {tempOriginsTwo.length, reinterpret_cast<ShapeDescriptor::OrientedPoint *>(tempOriginsTwo.content)};

        ShapeDescriptor::gpu::PointCloud pointCloudOne = ShapeDescriptor::utilities::sampleMesh(deviceMeshOne, sampleCount, randomSeed);
        ShapeDescriptor::gpu::PointCloud pointCloudTwo = ShapeDescriptor::utilities::sampleMesh(deviceMeshTwo, sampleCount, randomSeed);

        ShapeDescriptor::gpu::array<ShapeDescriptor::ShapeContextDescriptor> dOne =
            ShapeDescriptor::gpu::generate3DSCDescriptors(pointCloudOne, deviceSpinOriginsOne, pointDensityRadius, minSupportRadius, maxSupportRadius);
        descriptorsOne = ShapeDescriptor::copy::deviceArrayToHost(dOne);

        ShapeDescriptor::gpu::array<ShapeDescriptor::ShapeContextDescriptor> dTwo =
            ShapeDescriptor::gpu::generate3DSCDescriptors(pointCloudTwo, deviceSpinOriginsTwo, pointDensityRadius, minSupportRadius, maxSupportRadius);
        descriptorsTwo = ShapeDescriptor::copy::deviceArrayToHost(dTwo);

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

    std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor>> descriptors;
    descriptors[0] = descriptorsOne;
    descriptors[1] = descriptorsTwo;

    ShapeDescriptor::free::array(spinOriginsOne);
    ShapeDescriptor::free::array(spinOriginsTwo);

    return descriptors;
}