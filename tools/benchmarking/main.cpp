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
#include <variant>
#include <map>
#include <tsl/ordered_map.h>
#include <json.hpp>
#include <ctime>

template <class Key, class T, class Ignore, class Allocator,
          class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>,
          class AllocatorPair = typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, T>>,
          class ValueTypeContainer = std::vector<std::pair<Key, T>, AllocatorPair>>

using ordered_map = tsl::ordered_map<Key, T, Hash, KeyEqual, AllocatorPair, ValueTypeContainer>;

using json = nlohmann::basic_json<ordered_map>;

using descriptorType = std::variant<
    std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>>,
    std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor>>,
    std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor>>>;

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

std::vector<std::variant<int, std::string>> prepareMetadata(std::filesystem::path metadataPath)
{
    if (metadataPath != "")
    {
        return generateMetadata(metadataPath);
    }

    return std::vector<std::variant<int, std::string>>();
}

double singleObjectBenchmark(ShapeDescriptor::cpu::Mesh meshOne, ShapeDescriptor::cpu::Mesh meshTwo, std::vector<std::variant<int, std::string>> metadata, int algorithm, std::string hardware)
{
    descriptorType descriptors;
    double similarity = 0;

    switch (algorithm)
    {
    case 0:
    {
        descriptors = Benchmarking::utilities::descriptor::generateRICIDescriptors(meshOne, meshTwo, metadata, hardware);

        ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> dOne = std::get<std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>>>(descriptors)[0];
        ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> dTwo = std::get<std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>>>(descriptors)[1];

        similarity = Benchmarking::utilities::distance::cosineSimilarityBetweenTwoDescriptors(dOne, dTwo, metadata);

        ShapeDescriptor::free::array(dOne);
        ShapeDescriptor::free::array(dTwo);

        break;
    }
    case 1:
    {
        descriptors = Benchmarking::utilities::descriptor::generateQUICCIDescriptors(meshOne, meshTwo, metadata, hardware);

        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> dOne = std::get<std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor>>>(descriptors)[0];
        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> dTwo = std::get<std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor>>>(descriptors)[1];

        similarity = Benchmarking::utilities::distance::cosineSimilarityBetweenTwoDescriptors(dOne, dTwo, metadata);

        ShapeDescriptor::free::array(dOne);
        ShapeDescriptor::free::array(dTwo);

        break;
    }
    case 2:
    {
        descriptors = Benchmarking::utilities::descriptor::generateSpinImageDescriptors(meshOne, meshTwo, metadata);

        ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> dOne = std::get<std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor>>>(descriptors)[0];
        ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> dTwo = std::get<std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor>>>(descriptors)[1];

        similarity = Benchmarking::utilities::distance::cosineSimilarityBetweenTwoDescriptors(dOne, dTwo, metadata);

        ShapeDescriptor::free::array(dOne);
        ShapeDescriptor::free::array(dTwo);

        break;
    }
    default:
    {
        descriptors = Benchmarking::utilities::descriptor::generateRICIDescriptors(meshOne, meshTwo, metadata, hardware);

        ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> dOne = std::get<std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>>>(descriptors)[0];
        ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> dTwo = std::get<std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>>>(descriptors)[1];

        similarity = Benchmarking::utilities::distance::cosineSimilarityBetweenTwoDescriptors(dOne, dTwo, metadata);

        ShapeDescriptor::free::array(dOne);
        ShapeDescriptor::free::array(dTwo);
    }
    };

    return similarity;
}

json multipleObjectsBenchmark(std::string objectsFolder, std::string originalsFolderName, json jsonOutput, std::string hardware)
{
    std::vector<std::string> folders;
    std::string originalObjectFolderPath;

    jsonOutput["runDate"] = std::time(0);

    for (auto &p : std::filesystem::directory_iterator(objectsFolder))
    {
        if (p.is_directory())
        {
            if (originalObjectFolderPath.empty() && p.path().string().substr(p.path().string().find_last_of("/") + 1) == originalsFolderName)
            {
                originalObjectFolderPath = p.path().string();
            }
            else
            {
                folders.push_back(p.path().string());
            }
        }
    }

    std::map<int, std::string> descriptorAlgorithms; // Expand later
    descriptorAlgorithms[0] = "RICI";
    descriptorAlgorithms[1] = "QUICCI";

    for (auto &folder : std::filesystem::directory_iterator(originalObjectFolderPath))
    {
        std::string file = folder.path().string().substr(folder.path().string().find_last_of("/") + 1);
        int fileExtensionPlacement = file.find_last_of(".");
        std::string fileType = file.substr(fileExtensionPlacement + 1);
        std::string fileName = file.substr(0, fileExtensionPlacement);

        if (fileType == "obj")
        {
            std::cout << fileName << std::endl; // Might have to be changed if the folder ends up consisting of folders
            for (std::string folder : folders)
            {
                std::filesystem::path originalObjectPath;
                std::filesystem::path comparisonObjectPath;
                std::vector<std::variant<int, std::string>> metadata;

                ShapeDescriptor::cpu::Mesh meshOne;
                ShapeDescriptor::cpu::Mesh meshTwo;

                std::string comparisonFolder = folder + "/" + fileName;

                try
                {
                    originalObjectPath = originalObjectFolderPath + "/" + fileName + ".obj";
                    comparisonObjectPath = comparisonFolder + "/" + fileName + ".obj";
                    metadata = prepareMetadata(comparisonFolder + "/" + fileName + ".txt");

                    meshOne = ShapeDescriptor::utilities::loadMesh(originalObjectPath, true);
                    meshTwo = ShapeDescriptor::utilities::loadMesh(comparisonObjectPath, true);
                }
                catch (const std::exception e)
                {
                    std::cout << "Comparison object " << comparisonObjectPath << " not found..." << std::endl;
                    continue;
                }

                for (auto d : descriptorAlgorithms)
                {
                    int start = std::time(0);
                    double sim = singleObjectBenchmark(meshOne, meshTwo, metadata, d.first, hardware);
                    int end = std::time(0);

                    std::cout << "Similarity: " << sim << std::endl;
                    jsonOutput["results"][fileName][d.second]["similarity"] = sim;
                    jsonOutput["results"][fileName][d.second]["time"] = end - start;
                }

                ShapeDescriptor::free::mesh(meshOne);
                ShapeDescriptor::free::mesh(meshTwo);
            }
        }
    }

    return jsonOutput;
}

int main(int argc, const char **argv)
{
    arrrgh::parser parser("benchmarking", "Compare how similar two objects are (only OBJ file support)");
    const auto &originalObject = parser.add<std::string>("original-object", "Original object.", 'o', arrrgh::Optional, "");
    const auto &comparisonObject = parser.add<std::string>("comparison-object", "Object to compare to the original.", 'c', arrrgh::Optional, "");
    const auto &objectsFolder = parser.add<std::string>("objects-folder", "Folder consisting of sub-directories with all the different objects and their metadata", 'f', arrrgh::Optional, "");
    const auto &originalsFolderName = parser.add<std::string>("originals-folder", "Folder name with all the original objects (for example, RecalculatedNormals)", 'n', arrrgh::Optional, "RecalculatedNormals");
    const auto &metadataPath = parser.add<std::string>("metadata", "Path to metadata describing which vertecies that are changed", 'm', arrrgh::Optional, "");
    const auto &outputPath = parser.add<std::string>("output-path", "Path to the output", 'p', arrrgh::Optional, "");
    const auto &descriptorAlgorithm = parser.add<int>("descriptor-algorithm", "Which descriptor algorithm to use [0 for radial-intersection-count-images, 1 for quick-intersection-count-change-images ...will add more:)]", 'a', arrrgh::Optional, 0);
    const auto &distanceAlgorithm = parser.add<int>("distance-algorithm", "Which distance algorithm to use [0 for euclidian, ...will add more:)]", 'd', arrrgh::Optional, 0);
    const auto &hardware = parser.add<std::string>("hardware-type", "cpu or gpu", 't', arrrgh::Optional, "cpu");
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

    json jsonOutput;

    std::vector<std::variant<int, std::string>> metadata = prepareMetadata(metadataPath.value());

    if (originalObject.value() != "" && comparisonObject.value() != "" && (objectsFolder.value() == "" && originalsFolderName.value() == ""))
    {
        std::cout << "Comparing two objects..." << std::endl;
        std::filesystem::path objectOne = originalObject.value();
        std::filesystem::path objectTwo = comparisonObject.value();

        ShapeDescriptor::cpu::Mesh meshOne = ShapeDescriptor::utilities::loadMesh(objectOne, true);
        ShapeDescriptor::cpu::Mesh meshTwo = ShapeDescriptor::utilities::loadMesh(objectTwo, true);

        double similarity = singleObjectBenchmark(meshOne, meshTwo, metadata, descriptorAlgorithm.value(), hardware.value());

        std::cout << similarity << std::endl;

        ShapeDescriptor::free::mesh(meshOne);
        ShapeDescriptor::free::mesh(meshTwo);
    }
    else if (objectsFolder.value() != "" && originalsFolderName.value() != "" && (originalObject.value() == "" && comparisonObject.value() == ""))
    {
        std::cout << "Comparing all objects in folder..." << std::endl;
        jsonOutput = multipleObjectsBenchmark(objectsFolder.value(), originalsFolderName.value(), jsonOutput, hardware.value());

        std::ofstream outFile(outputPath.value());
        outFile << jsonOutput.dump(4);
        outFile.close();

        std::cout
            << "Results stored to file" << std::endl;
    }
    else
    {
        std::cout << "Wrong inputs, exiting..." << std::endl;
    }

    return 0;
}