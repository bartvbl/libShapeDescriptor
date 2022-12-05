#include "cosine.h"
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <math.h>

double cosineSimilarity(ShapeDescriptor::RICIDescriptor dOne, ShapeDescriptor::RICIDescriptor dTwo)
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

double Benchmarking::utilities::distance::cosineSimilarityBetweenTwoDescriptors(ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptorsOne, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptorsTwo)
{
    int index = 0;
    double sumOfSimilarities = 0;

    while (index < descriptorsOne.length && index < descriptorsTwo.length)
    {
        sumOfSimilarities += cosineSimilarity(descriptorsOne.content[1], descriptorsTwo.content[1]);
        index++;
    }

    int longestLength = (descriptorsOne.length > descriptorsTwo.length) ? descriptorsOne.length : descriptorsTwo.length;
    double averageSimilarity = sumOfSimilarities / longestLength;

    return averageSimilarity;
}