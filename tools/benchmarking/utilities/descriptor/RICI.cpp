#include "RICI.h"
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <vector>

std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>> Benchmarking::utilities::descriptor::similarityBetweenTwoObjectsWithRICI(ShapeDescriptor::cpu::Mesh meshOne, ShapeDescriptor::cpu::Mesh meshTwo, std::vector<std::variant<int, std::string>> metadata)
{
    float supportRadius = 1;
    std::cout << "Generating Radial Intersection Count Images" << std::endl;

    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOriginsOne = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(meshOne);
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOriginsTwo = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(meshTwo);

    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptorsOne = ShapeDescriptor::cpu::generateRadialIntersectionCountImages(meshOne, spinOriginsOne, supportRadius);
    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptorsTwo = ShapeDescriptor::cpu::generateRadialIntersectionCountImages(meshTwo, spinOriginsTwo, supportRadius);

    std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>> descriptors;
    descriptors[0] = descriptorsOne;
    descriptors[1] = descriptorsTwo;

    ShapeDescriptor::free::array(spinOriginsOne);
    ShapeDescriptor::free::array(spinOriginsTwo);
    ShapeDescriptor::free::array(descriptorsOne);
    ShapeDescriptor::free::array(descriptorsTwo);

    return descriptors;
}
