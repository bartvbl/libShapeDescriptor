#pragma once
#include <benchmarking/utilities/descriptor/RICI.h>
#include <benchmarking/utilities/descriptor/QUICCI.h>
#include <benchmarking/utilities/descriptor/spinImage.h>
#include <benchmarking/utilities/descriptor/3dShapeContext.h>
#include <benchmarking/utilities/descriptor/FPFH.h>
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
                                                   int distanceFunction)
            {
                std::cout << "Calculating the similarity of the two objects" << std::endl
                          << std::flush;

                // For rotatating shape contexts
                if constexpr (std::is_same_v<T, ShapeDescriptor::ShapeContextDescriptor>)
                {
                    double bestSimilarity = 0;

                    double (*similarityFunctionOffset)(T, T, int);

                    switch (distanceFunction)
                    {
                    case 0:
                    {
                        similarityFunctionOffset = &cosineSimilarityOffset;
                        std::cout << "Using Cosine Similarity" << std::endl;
                        break;
                    }
                    case 1:
                    {
                        similarityFunctionOffset = &euclidianSimilarityOffset;
                        std::cout << "Using Euclidian Similarity" << std::endl;
                        break;
                    }
                    default:
                    {
                        std::cout << "Invalid distance function specified. Defaulting to Euclidian Similarity" << std::endl;
                        similarityFunctionOffset = &euclidianSimilarityOffset;
                        break;
                    }
                    }

                    for (int sliceOffset = 0; sliceOffset < SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT - 1; sliceOffset++)
                    {
                        double similaritySum = 0;

                        for (int i = 0; i < descriptorsOne.length; i++)
                        {
                            ShapeDescriptor::ShapeContextDescriptor dOne = descriptorsOne.content[i];
                            ShapeDescriptor::ShapeContextDescriptor dTwo = descriptorsTwo.content[i];

                            similaritySum += similarityFunctionOffset(dOne, dTwo, sliceOffset);
                        }

                        double avgSimilarity = similaritySum / descriptorsOne.length;

                        bestSimilarity = std::max(bestSimilarity, avgSimilarity);
                    }

                    return bestSimilarity;
                }

                // Rest of the descriptors
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

                for (int i = 0; i < descriptorsOne.length; i++)
                {
                    sumOfSimilarities += similarityFunction(descriptorsOne.content[i], descriptorsTwo.content[i]);
                }

                double averageSimilarity = sumOfSimilarities / descriptorsOne.length;

                return averageSimilarity;
            }
        }
    }
}
