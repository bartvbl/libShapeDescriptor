#include "cosine.h"
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <benchmarking/utilities/distance/generateFakeMetadata.h>
#include <math.h>
#include <vector>

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

double Benchmarking::utilities::distance::cosineSimilarityBetweenTwoDescriptors(ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptorsOne, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptorsTwo, std::vector<std::variant<int, std::string>> metadata)
{
    double sumOfSimilarities = 0;
    if (metadata.size() == 0)
    {
        metadata = generateFakeMetadata(descriptorsOne.length);
    }

    for (int i = 0; i < metadata.size(); i++)
    {
        try
        {
            int index = std::get<int>(metadata.at(i));
            sumOfSimilarities += cosineSimilarity(descriptorsOne.content[i], descriptorsTwo.content[index]);
        }
        catch (std::exception e)
        {
            continue;
        }
    }

    double averageSimilarity = sumOfSimilarities / metadata.size();

    return averageSimilarity;
}