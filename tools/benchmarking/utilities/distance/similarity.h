#pragma once
#include <benchmarking/utilities/descriptor/RICI.h>
#include <benchmarking/utilities/descriptor/QUICCI.h>
#include <benchmarking/utilities/descriptor/spinImage.h>
#include <benchmarking/utilities/descriptor/3dShapeContext.h>
#include <benchmarking/utilities/descriptor/FPFH.h>
#include <benchmarking/utilities/distance/generateFakeMetadata.h>
#include <benchmarking/utilities/distance/cosine.h>
#include <benchmarking/utilities/distance/euclidian.h>
#include <vector>
#include <variant>
#include <shapeDescriptor/cpu/types/array.h>

namespace Benchmarking
{
    namespace utilities
    {
        namespace distance
        {
            template <typename T>
            double similarityBetweenTwoDescriptors(ShapeDescriptor::cpu::array<T> descriptorsOne,
                                                   ShapeDescriptor::cpu::array<T> descriptorsTwo,
                                                   std::vector<std::variant<int, std::string>> metadata,
                                                   int distanceFunction)
            {
                std::cout << "Calculating the similarity of the two objects" << std::endl
                          << std::flush;

                double sumOfSimilarities = 0;

                double (*similarityFunction)(T, T);
                switch (distanceFunction)
                {
                case 0:
                {
                    similarityFunction = &cosineSimilarity;
                    std::cout << "Using Cosine Similarity" << std::endl;
                    break;
                }
                case 1:
                {
                    similarityFunction = &euclidianSimilarity;
                    std::cout << "Using Euclidian Similarity" << std::endl;
                    break;
                }
                default:
                {
                    similarityFunction = &cosineSimilarity;
                    std::cout << "Using Cosine Similarity" << std::endl;
                    break;
                }
                }

                for (int i = 0; i < metadata.size(); i++)
                {
                    try
                    {
                        int index = std::get<int>(metadata.at(i));
                        sumOfSimilarities += similarityFunction(descriptorsOne.content[i], descriptorsTwo.content[index]);
                    }
                    catch (std::exception e)
                    {
                        continue;
                    }
                }

                double averageSimilarity = sumOfSimilarities / metadata.size();

                return averageSimilarity;
            }
        }
    }
}