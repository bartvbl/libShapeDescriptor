#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/read/MeshLoader.h>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <shapeDescriptor/cpu/quickIntersectionCountImageGenerator.h>
#include <shapeDescriptor/cpu/spinImageGenerator.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <benchmarking/utilities/descriptor/RICI.h>
#include <benchmarking/utilities/descriptor/QUICCI.h>
#include <benchmarking/utilities/descriptor/spinImage.h>
#include <benchmarking/utilities/distance/cosine.h>
#include <iostream>
#include <fstream>
#include <arrrgh.hpp>
#include <vector>
#include <map>

using descriptorType = std::variant<std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>>, std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor>>, std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor>>>;

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
    const auto &descriptorAlgorithm = parser.add<int>("descriptor-algorithm", "Which descriptor algorithm to use [0 for radial-intersection-count-images, 1 for quick-intersection-count-change-images ...will add more:)]", 'a', arrrgh::Optional, 0);
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
        std::cout << "No metadata provided, generating metadata..." << std::endl;
    }
    else
    {
        metadata = generateMetadata(metadataPath.value());
    }

    ShapeDescriptor::cpu::Mesh meshOne = ShapeDescriptor::utilities::loadMesh(objectOne, true);
    ShapeDescriptor::cpu::Mesh meshTwo = ShapeDescriptor::utilities::loadMesh(objectTwo, true);

    descriptorType descriptors;
    double similarity = 0;

    switch (descriptorAlgorithm.value())
    {
    case 0:
        descriptors = Benchmarking::utilities::descriptor::similarityBetweenTwoObjectsWithRICI(meshOne, meshTwo, metadata);
        similarity = Benchmarking::utilities::distance::cosineSimilarityBetweenTwoDescriptors(std::get<std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>>>(descriptors)[0], std::get<std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>>>(descriptors)[1], metadata);
        break;
    case 1:
        descriptors = Benchmarking::utilities::descriptor::similarityBetweenTwoObjectsWithQUICCI(meshOne, meshTwo, metadata);
        similarity = Benchmarking::utilities::distance::cosineSimilarityBetweenTwoDescriptors(std::get<std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor>>>(descriptors)[0], std::get<std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor>>>(descriptors)[1], metadata);
        break;
    case 2:
        descriptors = Benchmarking::utilities::descriptor::similarityBetweenTwoObjectsWithSpinImage(meshOne, meshTwo, metadata);
        similarity = Benchmarking::utilities::distance::cosineSimilarityBetweenTwoDescriptors(std::get<std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor>>>(descriptors)[0], std::get<std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor>>>(descriptors)[1], metadata);
        break;
    default:
        descriptors = Benchmarking::utilities::descriptor::similarityBetweenTwoObjectsWithRICI(meshOne, meshTwo, metadata);
        similarity = Benchmarking::utilities::distance::cosineSimilarityBetweenTwoDescriptors(std::get<std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>>>(descriptors)[0], std::get<std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>>>(descriptors)[1], metadata);
    };

    std::cout << similarity << std::endl;

    ShapeDescriptor::free::mesh(meshOne);
    ShapeDescriptor::free::mesh(meshTwo);
}