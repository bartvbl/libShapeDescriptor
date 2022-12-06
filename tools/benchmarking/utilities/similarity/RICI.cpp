#include "RICI.h"
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <benchmarking/utilities/distance/cosine.h>
#include <vector>

double Benchmarking::utilities::similarity::similarityBetweenTwoObjectsWithRICI(ShapeDescriptor::cpu::Mesh meshOne, ShapeDescriptor::cpu::Mesh meshTwo, double (*distanceAlgorithm)(ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>, std::vector<std::variant<int, std::string>>), std::vector<std::variant<int, std::string>> metadata)
{
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOriginsOne = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(meshOne);
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOriginsTwo = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(meshTwo);

    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptorsOne = ShapeDescriptor::cpu::generateRadialIntersectionCountImages(meshOne, spinOriginsOne, 1);
    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptorsTwo = ShapeDescriptor::cpu::generateRadialIntersectionCountImages(meshTwo, spinOriginsTwo, 1);

    double averageSimilarity = distanceAlgorithm(descriptorsOne, descriptorsTwo, metadata);

    ShapeDescriptor::free::array(spinOriginsOne);
    ShapeDescriptor::free::array(spinOriginsTwo);
    ShapeDescriptor::free::array(descriptorsOne);
    ShapeDescriptor::free::array(descriptorsTwo);

    return averageSimilarity;
}