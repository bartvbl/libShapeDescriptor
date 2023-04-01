#pragma once
#include <vector>
#include <variant>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <shapeDescriptor/cpu/quickIntersectionCountImageGenerator.h>
#include <shapeDescriptor/cpu/spinImageGenerator.h>
#include <shapeDescriptor/gpu/fastPointFeatureHistogramGenerator.cuh>
#include <shapeDescriptor/gpu/3dShapeContextSearcher.cuh>

namespace Benchmarking
{
    namespace utilities
    {
        namespace metadata
        {
            int getNumberOfDeletedVerticiesInMetadata(std::vector<std::variant<int, std::string>> metadata)
            {
                int numberOfDeletedVerticies = 0;
                for (int i = 0; i < metadata.size(); i++)
                {
                    try
                    {
                        std::get<int>(metadata[i]);
                    }
                    catch (std::exception &e)
                    {
                        numberOfDeletedVerticies++;
                    }
                }
                return numberOfDeletedVerticies;
            }

            // Need to be implemented in header, as C++ requires it
            template <typename T>
            std::vector<ShapeDescriptor::cpu::array<T>> transformDescriptorsToMatchMetadata(
                ShapeDescriptor::cpu::array<T> descriptorOriginal,
                ShapeDescriptor::cpu::array<T> descriptorComparison,
                std::vector<std::variant<int, std::string>> metadata)
            {
                int numberOfDeletedVerticies = getNumberOfDeletedVerticiesInMetadata(metadata);
                int realLengthOfDescriptor = metadata.size() - numberOfDeletedVerticies;

                ShapeDescriptor::cpu::array<T> transformedOriginalDescriptor = ShapeDescriptor::cpu::array<T>(realLengthOfDescriptor);
                ShapeDescriptor::cpu::array<T> transformedComparisonDescriptor = ShapeDescriptor::cpu::array<T>(realLengthOfDescriptor);

                int metadataIndex = 0;
                int descriptorIndex = 0;

                while (metadataIndex < realLengthOfDescriptor)
                {
                    int comparsisonIndex;
                    try
                    {
                        comparsisonIndex = std::get<int>(metadata[descriptorIndex]);
                    }
                    catch (std::exception &e)
                    {
                        descriptorIndex++;
                        continue;
                    }

                    transformedOriginalDescriptor.content[metadataIndex] = descriptorOriginal.content[metadataIndex];
                    transformedComparisonDescriptor.content[metadataIndex] = descriptorComparison.content[comparsisonIndex];

                    descriptorIndex++;
                    metadataIndex++;
                }

                std::vector<ShapeDescriptor::cpu::array<T>> outputDescriptors;
                outputDescriptors.push_back(transformedOriginalDescriptor);
                outputDescriptors.push_back(transformedComparisonDescriptor);

                return outputDescriptors;
            }
        }
    }
}