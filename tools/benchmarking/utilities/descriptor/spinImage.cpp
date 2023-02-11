#include "spinImage.h"
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/cpu/spinImageGenerator.h>
#include <shapeDescriptor/utilities/meshSampler.h>
#include <vector>
#include <map>

std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor>> Benchmarking::utilities::descriptor::similarityBetweenTwoObjectsWithSpinImage(ShapeDescriptor::cpu::Mesh meshOne, ShapeDescriptor::cpu::Mesh meshTwo, std::vector<std::variant<int, std::string>> metadata)
{
    std::cout << "Generating Spin Images" << std::endl;
    size_t sampleCount = 1000000;
    size_t randomSeed = 5553580318008;
    float supportRadius = 0.1f;
    float supportAngleDegrees = 10.0f;

    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOriginsOne = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(meshOne);
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOriginsTwo = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(meshTwo);

    ShapeDescriptor::cpu::PointCloud pointCloudOne = ShapeDescriptor::utilities::sampleMesh(meshOne, sampleCount, randomSeed);
    ShapeDescriptor::cpu::PointCloud pointCloudTwo = ShapeDescriptor::utilities::sampleMesh(meshTwo, sampleCount, randomSeed);

    ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> descriptorsOne = ShapeDescriptor::cpu::generateSpinImages(pointCloudOne, spinOriginsOne, supportRadius, supportAngleDegrees);
    ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> descriptorsTwo = ShapeDescriptor::cpu::generateSpinImages(pointCloudTwo, spinOriginsTwo, supportRadius, supportAngleDegrees);

    std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor>> descriptors;
    descriptors[0] = descriptorsOne;
    descriptors[1] = descriptorsTwo;

    ShapeDescriptor::free::array(spinOriginsOne);
    ShapeDescriptor::free::array(spinOriginsTwo);
    ShapeDescriptor::free::array(descriptorsOne);
    ShapeDescriptor::free::array(descriptorsTwo);

    return descriptors;
}
