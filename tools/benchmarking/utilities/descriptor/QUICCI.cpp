#include "QUICCI.h"
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/cpu/quickIntersectionCountImageGenerator.h>
#include <benchmarking/utilities/distance/cosine.h>
#include <vector>

std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor>> Benchmarking::utilities::descriptor::similarityBetweenTwoObjectsWithQUICCI(ShapeDescriptor::cpu::Mesh meshOne, ShapeDescriptor::cpu::Mesh meshTwo, std::vector<std::variant<int, std::string>> metadata)
{
    std::cout << "Generating Quick Intersection Count Change Images" << std::endl;

    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOriginsOne = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(meshOne);
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOriginsTwo = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(meshTwo);

    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptorsOne = ShapeDescriptor::cpu::generateQUICCImages(meshOne, spinOriginsOne, 0.3f);
    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptorsTwo = ShapeDescriptor::cpu::generateQUICCImages(meshTwo, spinOriginsTwo, 0.3f);

    std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor>> descriptors;
    descriptors[0] = descriptorsOne;
    descriptors[1] = descriptorsTwo;

    ShapeDescriptor::free::array(spinOriginsOne);
    ShapeDescriptor::free::array(spinOriginsTwo);
    ShapeDescriptor::free::array(descriptorsOne);
    ShapeDescriptor::free::array(descriptorsTwo);

    return descriptors;
}