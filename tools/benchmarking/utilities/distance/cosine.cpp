#include "cosine.h"
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <shapeDescriptor/cpu/quickIntersectionCountImageGenerator.h>
#include <shapeDescriptor/cpu/spinImageGenerator.h>
#include <benchmarking/utilities/distance/generateFakeMetadata.h>
#include <math.h>
#include <vector>
#include <iostream>

// RICI
double cosineSimilarity(ShapeDescriptor::RICIDescriptor dOne, ShapeDescriptor::RICIDescriptor dTwo)
{
    double dot = 0;
    double denominationA = 0;
    double denominationB = 0;

    for (int i = 0; i < spinImageWidthPixels * spinImageWidthPixels; i++)
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
    std::cout << "Calculating the Cosine Similarity of the two objects" << std::endl
              << std::flush;

    double sumOfSimilarities = 0;

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

    std::cout << std::endl;

    double averageSimilarity = sumOfSimilarities / metadata.size();

    return averageSimilarity;
}

// QUICCI
double cosineSimilarity(ShapeDescriptor::QUICCIDescriptor dOne, ShapeDescriptor::QUICCIDescriptor dTwo)
{
    double dot = 0;
    double denominationA = 0;
    double denominationB = 0;

    for (int i = 0; i < ShapeDescriptor::QUICCIDescriptorLength; i++)
    {
        dot += (double)dOne.contents[i] * (double)dTwo.contents[i];
        denominationA += pow((double)dOne.contents[i], 2);
        denominationB += pow((double)dTwo.contents[i], 2);
    }

    double similarity = dot / sqrt(denominationA * denominationB);

    return isnan(similarity) ? 0 : similarity;
}

double Benchmarking::utilities::distance::cosineSimilarityBetweenTwoDescriptors(ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptorsOne, ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptorsTwo, std::vector<std::variant<int, std::string>> metadata)
{
    std::cout << "Calculating the Cosine Similarity of the two objects" << std::endl
              << std::flush;

    double sumOfSimilarities = 0;

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

    std::cout << std::endl;

    double averageSimilarity = sumOfSimilarities / metadata.size();

    return averageSimilarity;
}

// Spin
double cosineSimilarity(ShapeDescriptor::SpinImageDescriptor dOne, ShapeDescriptor::SpinImageDescriptor dTwo)
{
    double dot = 0;
    double denominationA = 0;
    double denominationB = 0;

    for (int i = 0; i < ShapeDescriptor::QUICCIDescriptorLength; i++)
    {
        dot += (double)dOne.contents[i] * (double)dTwo.contents[i];
        denominationA += pow((double)dOne.contents[i], 2);
        denominationB += pow((double)dTwo.contents[i], 2);
    }

    double similarity = dot / sqrt(denominationA * denominationB);

    return isnan(similarity) ? 0 : similarity;
}

double Benchmarking::utilities::distance::cosineSimilarityBetweenTwoDescriptors(ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> descriptorsOne, ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> descriptorsTwo, std::vector<std::variant<int, std::string>> metadata)
{
    std::cout << "Calculating the Cosine Similarity of the two objects" << std::endl
              << std::flush;

    double sumOfSimilarities = 0;

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

    std::cout << std::endl;

    double averageSimilarity = sumOfSimilarities / metadata.size();

    return averageSimilarity;
}
