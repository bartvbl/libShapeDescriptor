#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/read/MeshLoader.h>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <iostream>

double cosineSimilarityBetweenTwoDescriptors(ShapeDescriptor::RICIDescriptor dOne, ShapeDescriptor::RICIDescriptor dTwo)
{
    double dot = 0;
    double denominationA = 0;
    double denominationB = 0;

    for (int i = 0; i < 1024; i++)
    {
        dot += dOne.contents[i] * dTwo.contents[i];
        denominationA += pow(dOne.contents[i], 2);
        denominationB += pow(dTwo.contents[i], 2);
    }

    double similarity = dot / sqrt(denominationA * denominationB);

    return isnan(similarity) ? 0 : similarity;
}

int main(int argc, const char **argv)
{
    float supportRadius = 1;

    std::filesystem::path objectOne = "/Users/jonathanbrooks/masteroppgaven/objects/shark.obj";
    std::filesystem::path objectTwo = "/Users/jonathanbrooks/masteroppgaven/objects/model.obj";

    ShapeDescriptor::cpu::Mesh meshOne = ShapeDescriptor::utilities::loadMesh(objectOne, true);
    ShapeDescriptor::cpu::Mesh meshTwo = ShapeDescriptor::utilities::loadMesh(objectTwo, true);

    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOriginsOne = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(meshOne);
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOriginsTwo = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(meshTwo);

    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptorsOne = ShapeDescriptor::cpu::generateRadialIntersectionCountImages(meshOne, spinOriginsOne, supportRadius);
    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptorsTwo = ShapeDescriptor::cpu::generateRadialIntersectionCountImages(meshTwo, spinOriginsTwo, supportRadius);

    int index = 0;
    double sumOfSimilarities = 0;

    while (index < descriptorsOne.length && index < descriptorsTwo.length)
    {
        sumOfSimilarities += cosineSimilarityBetweenTwoDescriptors(descriptorsOne.content[index], descriptorsTwo.content[index]);
        index++;
    }

    int longestLength = (descriptorsOne.length > descriptorsTwo.length) ? descriptorsOne.length : descriptorsTwo.length;
    double averageSimilarity = sumOfSimilarities / longestLength;

    std::cout << averageSimilarity * 100 << "%" << std::endl;

    ShapeDescriptor::free::mesh(meshOne);
    ShapeDescriptor::free::mesh(meshTwo);
    ShapeDescriptor::free::array(spinOriginsOne);
    ShapeDescriptor::free::array(spinOriginsTwo);
    ShapeDescriptor::free::array(descriptorsOne);
    ShapeDescriptor::free::array(descriptorsTwo);
}