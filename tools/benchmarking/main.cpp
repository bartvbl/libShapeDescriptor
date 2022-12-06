#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/read/MeshLoader.h>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <benchmarking/utilities/similarity/RICI.h>
#include <benchmarking/utilities/distance/cosine.h>
#include <iostream>
#include <fstream>
#include <arrrgh.hpp>
#include <vector>

std::vector<std::variant<int, std::string>> generateMetadata(std::filesystem::path metadataPath)
{
    std::vector<std::variant<int, std::string>> metadata;
    std::ifstream metadataFile;
    std::string line;

    metadataFile.open(metadataPath);
    if (metadataFile.is_open())
    {
        while (getline(metadataFile, line))
        {
            try
            {
                metadata.push_back(stoi(line));
            }
            catch (std::exception e)
            {
                if (line != "")
                {
                    metadata.push_back(line);
                }
            }
        }
        metadataFile.close();
    }

    return metadata;
}

int main(int argc, const char **argv)
{
    arrrgh::parser parser("benchmarking", "Compare how similar two objects are (only OBJ file support)");
    const auto &originalObject = parser.add<std::string>("original-object", "Original object.", 'o', arrrgh::Required, "");
    const auto &comparisonObject = parser.add<std::string>("comparison-object", "Object to compare to the original.", 'c', arrrgh::Required, "");
    const auto &metadataPath = parser.add<std::string>("metadata", "Path to metadata describing which vertecies that are changed", 'm', arrrgh::Optional, "");
    const auto &descriptorAlgorithm = parser.add<int>("descriptor-algorithm", "Which descriptor algorithm to use [0 for radial-intersection-count-images, ...will add more:)]", 'a', arrrgh::Optional, 0);
    const auto &distanceAlgorithm = parser.add<int>("distance-algorithm", "Which distance algorithm to use [0 for euclidian, ...will add more:)]", 'd', arrrgh::Optional, 0);
    const auto &help = parser.add<bool>("help", "Show help", 'h', arrrgh::Optional, false);

    try
    {
        parser.parse(argc, argv);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        parser.show_usage(std::cerr);
        exit(1);
    }

    if (help.value())
    {
        return 0;
    }

    std::filesystem::path objectOne = originalObject.value();
    std::filesystem::path objectTwo = comparisonObject.value();

    std::vector<std::variant<int, std::string>> metadata;

    if (metadataPath.value() == "")
    {
        metadata = std::vector<std::variant<int, std::string>>();
    }
    else
    {
        metadata = generateMetadata(metadataPath.value());
    }

    ShapeDescriptor::cpu::Mesh meshOne = ShapeDescriptor::utilities::loadMesh(objectOne, true);
    ShapeDescriptor::cpu::Mesh meshTwo = ShapeDescriptor::utilities::loadMesh(objectTwo, true);

    double (*distance)(ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>, std::vector<std::variant<int, std::string>>);

    switch (distanceAlgorithm)
    {
    case 0:
        distance = &Benchmarking::utilities::distance::cosineSimilarityBetweenTwoDescriptors;
        break;
    default:
        distance = &Benchmarking::utilities::distance::cosineSimilarityBetweenTwoDescriptors;
    }

    double similarity;

    switch (descriptorAlgorithm.value())
    {
    case 0:
        similarity = Benchmarking::utilities::similarity::similarityBetweenTwoObjectsWithRICI(meshOne, meshTwo, distance, metadata);
        break;
    default:
        similarity = Benchmarking::utilities::similarity::similarityBetweenTwoObjectsWithRICI(meshOne, meshTwo, distance, metadata);
    };

    std::cout << similarity << std::endl;

    ShapeDescriptor::free::mesh(meshOne);
    ShapeDescriptor::free::mesh(meshTwo);
}