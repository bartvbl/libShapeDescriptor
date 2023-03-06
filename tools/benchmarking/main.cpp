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
#include <benchmarking/utilities/descriptor/3dShapeContext.h>
#include <benchmarking/utilities/descriptor/FPFH.h>
#include <benchmarking/utilities/distance/similarity.h>
#include <benchmarking/utilities/distance/generateFakeMetadata.h>
#include <iostream>
#include <fstream>
#include <arrrgh.hpp>
#include <vector>
#include <variant>
#include <map>
#include <json.hpp>
#include <ctime>
#include <chrono>
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <cuda_runtime_api.h>
#endif

using json = nlohmann::json;

json originalObjectsData;

using descriptorType = std::variant<
    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>,
    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor>,
    ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor>,
    ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor>,
    ShapeDescriptor::cpu::array<ShapeDescriptor::FPFHDescriptor>>;

std::map<int, std::string> descriptorAlgorithms = {
    {0, "RICI"},
    {1, "QUICCI"},
    {2, "SI"},
    {3, "3DSC"},
    {4, "FPFH"}};

std::map<int, std::string>
    distanceFunctions = {{0, "Cosine"}, {1, "Euclidian"}};

struct
{
    std::string name;
    int clockRate;
    int memory;
} GPUInfo;

int runDate;

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

std::vector<std::variant<int, std::string>> prepareMetadata(std::filesystem::path metadataPath, int length = 0)
{
    if (std::filesystem::exists(metadataPath))
    {
        return generateMetadata(metadataPath);
    }

    return Benchmarking::utilities::distance::generateFakeMetadata(length);
}

descriptorType generateDescriptorsForObject(ShapeDescriptor::cpu::Mesh mesh,
                                            int algorithm,
                                            std::string hardware,
                                            std::chrono::duration<double> &elapsedTime,
                                            float supportRadius = 0.5f,
                                            float supportAngleDegress = 10.0f,
                                            float pointDensityRadius = 0.4f,
                                            float minSupportRadius = 0.2f,
                                            float maxSupportRadius = 1.0f,
                                            size_t pointCloudSampleCount = 100000,
                                            size_t randomSeed = 133713375318008)
{
    descriptorType descriptor;

    switch (algorithm)
    {
    case 0:
    {
        descriptor = Benchmarking::utilities::descriptor::generateRICIDescriptor(
            mesh, hardware, supportRadius, elapsedTime);
        break;
    }
    case 1:
    {
        descriptor = Benchmarking::utilities::descriptor::generateQUICCIDescriptor(
            mesh, hardware, supportRadius, elapsedTime);
        break;
    }
    case 2:
    {
        descriptor = Benchmarking::utilities::descriptor::generateSpinImageDescriptor(
            mesh, hardware, supportRadius, supportAngleDegress, pointCloudSampleCount, randomSeed, elapsedTime);
        break;
    }
    case 3:
    {
        descriptor = Benchmarking::utilities::descriptor::generate3DShapeContextDescriptor(
            mesh, hardware, pointCloudSampleCount, randomSeed, pointDensityRadius, minSupportRadius, maxSupportRadius, elapsedTime);
        break;
    }
    case 4:
    {
        descriptor = Benchmarking::utilities::descriptor::generateFPFHDescriptor(
            mesh, hardware, supportRadius, pointCloudSampleCount, randomSeed, elapsedTime);
        break;
    }
    default:
    {
        descriptor = Benchmarking::utilities::descriptor::generateRICIDescriptor(
            mesh, hardware, supportRadius, elapsedTime);
        break;
    }
    };

    return descriptor;
}

template <typename T>
double calculateSimilarity(ShapeDescriptor::cpu::array<T> dOriginal, ShapeDescriptor::cpu::array<T> dComparison, std::vector<std::variant<int, std::string>> metadata, int distanceFunction, bool freeArray)
{
    double sim = Benchmarking::utilities::distance::similarityBetweenTwoDescriptors<T>(dOriginal, dComparison, metadata, distanceFunction);

    if (freeArray)
    {
        ShapeDescriptor::free::array(dOriginal);
        ShapeDescriptor::free::array(dComparison);
    }

    return sim;
}

void multipleObjectsBenchmark(std::string objectsFolder, std::string originalsFolderName, std::string jsonPath, std::string hardware, std::string compareFolder)
{
    std::vector<std::string> folders;
    std::string originalObjectFolderPath;

    // This is hard coded for now, as this fits how we have structured the folder. Should be edited if you want the code more dynamic:^)
    std::string originalObjectCategory = "0-100";

    std::map<std::string, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>> originalDescriptorsRICI;
    std::map<std::string, ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor>> originalDescriptorsQUICCI;
    std::map<std::string, ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor>> originalDescriptorsSI;
    std::map<std::string, ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor>> originalDescriptors3DSC;
    std::map<std::string, ShapeDescriptor::cpu::array<ShapeDescriptor::FPFHDescriptor>> originalDescriptorsFPFH;

    if (compareFolder == "")
    {
        for (auto &p : std::filesystem::directory_iterator(objectsFolder))
        {
            if (p.is_directory())
            {
                if (originalObjectFolderPath.empty() && p.path().string().substr(p.path().string().find_last_of("/") + 1) == originalsFolderName)
                {
                    originalObjectFolderPath = p.path().string() + "/" + originalObjectCategory;
                }
                else
                {
                    folders.push_back(p.path().string());
                }
            }
        }
    }
    else
    {
        folders.push_back(objectsFolder + compareFolder);
    }

    for (std::string folder : folders)
    {
        std::chrono::steady_clock::time_point timeStart = std::chrono::steady_clock::now();
        json jsonOutput;
        std::string comparisonFolderName = folder.substr(folder.find_last_of("/") + 1);

        for (auto &categoryPath : std::filesystem::directory_iterator(folder))
        {
            std::string category = categoryPath.path().string().substr(categoryPath.path().string().find_last_of("/") + 1);

            if (!categoryPath.is_directory())
                continue;

            for (auto &originalObject : std::filesystem::directory_iterator(originalObjectFolderPath))
            {
                std::string originalFolder = originalObject.path().string();
                std::string fileName = originalFolder.substr(originalFolder.find_last_of("/") + 1);

                std::filesystem::path originalObjectPath;
                std::filesystem::path comparisonObjectPath;
                std::vector<std::variant<int, std::string>> metadata;

                ShapeDescriptor::cpu::Mesh meshOriginal;
                ShapeDescriptor::cpu::Mesh meshComparison;

                std::string comparisonFolder = folder + "/" + category + "/" + fileName;

                try
                {
                    originalObjectPath = originalFolder + "/" + fileName + ".obj";
                    comparisonObjectPath = comparisonFolder + "/" + fileName + ".obj";

                    meshOriginal = ShapeDescriptor::utilities::loadMesh(originalObjectPath);

                    // Our GoogleDataset objects do not include any normals
                    if (comparisonFolderName == "GoogleDataset")
                    {
                        meshComparison = ShapeDescriptor::utilities::loadMesh(comparisonObjectPath, true);
                    }
                    else
                    {
                        meshComparison = ShapeDescriptor::utilities::loadMesh(comparisonObjectPath);
                    }

                    metadata = prepareMetadata(comparisonFolder + "/" + fileName + ".txt", meshOriginal.vertexCount);
                }
                catch (const std::exception e)
                {
                    std::cout << "Comparison object " << comparisonObjectPath << " not found..." << std::endl;
                    continue;
                }

                for (auto a : descriptorAlgorithms)
                {
                    std::chrono::duration<double> elapsedSecondsDescriptorComparison;

                    descriptorType comparisonObject = generateDescriptorsForObject(meshComparison, a.first, hardware, elapsedSecondsDescriptorComparison);

                    for (auto d : distanceFunctions)
                    {
                        bool freeArray = d.first == distanceFunctions.size() - 1;
                        double sim = 0;

                        std::chrono::duration<double> elapsedSecondsDescriptorOriginal;

                        std::chrono::steady_clock::time_point distanceTimeStart;
                        std::chrono::steady_clock::time_point distanceTimeEnd;

                        switch (a.first)
                        {
                        case 0:
                        {
                            ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> original;
                            if (auto search = originalDescriptorsRICI.find(fileName); search != originalDescriptorsRICI.end())
                                original = originalDescriptorsRICI[fileName];
                            else
                            {
                                original = std::get<0>(generateDescriptorsForObject(meshOriginal, a.first, hardware, elapsedSecondsDescriptorOriginal));
                                originalDescriptorsRICI.insert({fileName, original});
                                originalObjectsData["results"][fileName][a.second]["generationTime"] = elapsedSecondsDescriptorOriginal.count();
                                originalObjectsData["results"][fileName][originalsFolderName]["vertexCount"] = meshOriginal.vertexCount;
                            }

                            distanceTimeStart = std::chrono::steady_clock::now();
                            sim = calculateSimilarity<ShapeDescriptor::RICIDescriptor>(original, std::get<0>(comparisonObject), metadata, d.first, freeArray);
                            distanceTimeEnd = std::chrono::steady_clock::now();
                            break;
                        }
                        case 1:
                        {
                            ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> original;
                            if (auto search = originalDescriptorsQUICCI.find(fileName); search != originalDescriptorsQUICCI.end())
                                original = originalDescriptorsQUICCI[fileName];
                            else
                            {
                                original = std::get<1>(generateDescriptorsForObject(meshOriginal, a.first, hardware, elapsedSecondsDescriptorOriginal));
                                originalDescriptorsQUICCI.insert({fileName, original});
                                originalObjectsData["results"][fileName][a.second]["generationTime"] = elapsedSecondsDescriptorOriginal.count();
                                originalObjectsData["results"][fileName][originalsFolderName]["vertexCount"] = meshOriginal.vertexCount;
                            }

                            distanceTimeStart = std::chrono::steady_clock::now();
                            sim = calculateSimilarity<ShapeDescriptor::QUICCIDescriptor>(original, std::get<1>(comparisonObject), metadata, d.first, freeArray);
                            distanceTimeEnd = std::chrono::steady_clock::now();
                            break;
                        }
                        case 2:
                        {
                            ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> original;
                            if (auto search = originalDescriptorsSI.find(fileName); search != originalDescriptorsSI.end())
                                original = originalDescriptorsSI[fileName];
                            else
                            {
                                original = std::get<2>(generateDescriptorsForObject(meshOriginal, a.first, hardware, elapsedSecondsDescriptorOriginal));
                                originalDescriptorsSI.insert({fileName, original});
                                originalObjectsData["results"][fileName][a.second]["generationTime"] = elapsedSecondsDescriptorOriginal.count();
                                originalObjectsData["results"][fileName][originalsFolderName]["vertexCount"] = meshOriginal.vertexCount;
                            }

                            distanceTimeStart = std::chrono::steady_clock::now();
                            sim = calculateSimilarity<ShapeDescriptor::SpinImageDescriptor>(original, std::get<2>(comparisonObject), metadata, d.first, freeArray);
                            distanceTimeEnd = std::chrono::steady_clock::now();
                            break;
                        }
                        case 3:
                        {
                            ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> original;
                            if (auto search = originalDescriptors3DSC.find(fileName); search != originalDescriptors3DSC.end())
                                original = originalDescriptors3DSC[fileName];
                            else
                            {
                                original = std::get<3>(generateDescriptorsForObject(meshOriginal, a.first, hardware, elapsedSecondsDescriptorOriginal));
                                originalDescriptors3DSC[fileName] = original;
                                originalObjectsData["results"][fileName][a.second]["generationTime"] = elapsedSecondsDescriptorOriginal.count();
                                originalObjectsData["results"][fileName][originalsFolderName]["vertexCount"] = meshOriginal.vertexCount;
                            }

                            distanceTimeStart = std::chrono::steady_clock::now();
                            sim = calculateSimilarity<ShapeDescriptor::ShapeContextDescriptor>(original, std::get<3>(comparisonObject), metadata, d.first, freeArray);
                            distanceTimeEnd = std::chrono::steady_clock::now();
                            break;
                        }
                        case 4:
                        {
                            ShapeDescriptor::cpu::array<ShapeDescriptor::FPFHDescriptor> original;
                            if (auto search = originalDescriptorsFPFH.find(fileName); search != originalDescriptorsFPFH.end())
                                original = originalDescriptorsFPFH[fileName];
                            else
                            {
                                original = std::get<4>(generateDescriptorsForObject(meshOriginal, a.first, hardware, elapsedSecondsDescriptorOriginal));
                                originalDescriptorsFPFH.insert({fileName, original});
                                originalObjectsData["results"][fileName][a.second]["generationTime"] = elapsedSecondsDescriptorOriginal.count();
                                originalObjectsData["results"][fileName][originalsFolderName]["vertexCount"] = meshOriginal.vertexCount;
                            }

                            distanceTimeStart = std::chrono::steady_clock::now();
                            sim = calculateSimilarity<ShapeDescriptor::FPFHDescriptor>(original, std::get<4>(comparisonObject), metadata, d.first, freeArray);
                            distanceTimeEnd = std::chrono::steady_clock::now();
                            break;
                        }
                        default:
                        {
                            ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> original;
                            if (auto search = originalDescriptorsRICI.find(fileName); search != originalDescriptorsRICI.end())
                                original = originalDescriptorsRICI[fileName];
                            else
                            {
                                original = std::get<0>(generateDescriptorsForObject(meshOriginal, a.first, hardware, elapsedSecondsDescriptorOriginal));
                                originalDescriptorsRICI.insert({fileName, original});
                                originalObjectsData["results"][fileName][a.second]["generationTime"] = elapsedSecondsDescriptorOriginal.count();
                                originalObjectsData["results"][fileName][originalsFolderName]["vertexCount"] = meshOriginal.vertexCount;
                            }

                            distanceTimeStart = std::chrono::steady_clock::now();
                            sim = calculateSimilarity<ShapeDescriptor::RICIDescriptor>(original, std::get<0>(comparisonObject), metadata, d.first, freeArray);
                            distanceTimeEnd = std::chrono::steady_clock::now();
                            break;
                        }
                        }

                        std::chrono::duration<double> elapsedSecondsDistance = distanceTimeEnd - distanceTimeStart;

                        jsonOutput["results"][fileName][comparisonFolderName][a.second][category]["generationTime"] = elapsedSecondsDescriptorComparison.count();
                        jsonOutput["results"][fileName][comparisonFolderName][a.second][category][d.second]["similarity"] = (double)sim;
                        jsonOutput["results"][fileName][comparisonFolderName][a.second][category][d.second]["time"] = elapsedSecondsDistance.count();

                        jsonOutput["results"][fileName][comparisonFolderName]["vertexCount"] = meshComparison.vertexCount;
                    }
                }
                ShapeDescriptor::free::mesh(meshOriginal);
                ShapeDescriptor::free::mesh(meshComparison);
            }
        }
        std::chrono::steady_clock::time_point timeAfter = std::chrono::steady_clock::now();
        std::chrono::duration<double> totalRunTime = timeAfter - timeStart;

        jsonOutput["runDate"] = runDate;
        jsonOutput["hardware"]["type"] = hardware;
        jsonOutput["runTime"] = totalRunTime.count();

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
        jsonOutput["hardware"]["gpu"]["name"] = GPUInfo.name;
        jsonOutput["hardware"]["gpu"]["clockRate"] = GPUInfo.clockRate;
        jsonOutput["hardware"]["gpu"]["memory"] = GPUInfo.memory;
#endif

        std::string outputDirectory = jsonPath + std::to_string(runDate);

        std::filesystem::create_directory(outputDirectory);

        std::string outputFilePath = outputDirectory + "/" + comparisonFolderName + ".json";
        std::ofstream outFile(outputFilePath);
        outFile << jsonOutput.dump(4);
        outFile.close();

        std::cout << "Results stored to " << outputFilePath << std::endl;
    }
}

int main(int argc, const char **argv)
{
    arrrgh::parser parser("benchmarking", "Compare how similar two objects are (only OBJ file support)");
    const auto &originalObject = parser.add<std::string>("original-object", "Original object.", 'o', arrrgh::Optional, "");
    const auto &comparisonObject = parser.add<std::string>("comparison-object", "Object to compare to the original.", 'c', arrrgh::Optional, "");
    const auto &objectsFolder = parser.add<std::string>("objects-folder", "Folder consisting of sub-directories with all the different objects and their metadata", 'f', arrrgh::Optional, "");
    const auto &originalsFolderName = parser.add<std::string>("originals-folder", "Folder name with all the original objects (for example, RecalculatedNormals)", 'n', arrrgh::Optional, "RecalculatedNormals");
    const auto &compareFolder = parser.add<std::string>("compare-folder", "If you only want to compare the originals to a specific folder (for example, ObjectsWithHoles)", 'F', arrrgh::Optional, "");
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

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    cudaDeviceProp device_information;
    cudaGetDeviceProperties(&device_information, 0);
    GPUInfo.name = std::string(device_information.name);
    GPUInfo.clockRate = device_information.clockRate;
    GPUInfo.memory = device_information.totalGlobalMem / (1024 * 1024);
#endif

    if (originalObject.value() != "" && comparisonObject.value() != "")
    {
        int timeStart = std::time(0);
        std::filesystem::path objectOne = originalObject.value();
        std::filesystem::path objectTwo = comparisonObject.value();

        ShapeDescriptor::cpu::Mesh meshOne = ShapeDescriptor::utilities::loadMesh(objectOne);
        ShapeDescriptor::cpu::Mesh meshTwo = ShapeDescriptor::utilities::loadMesh(objectTwo, true);

        std::vector<std::variant<int, std::string>> metadata;

        if (metadataPath.value() == "")
        {
            metadata = prepareMetadata("", meshOne.vertexCount);
        }
        else
        {
            metadata = prepareMetadata(metadataPath.value());
        }

        std::chrono::duration<double> elapsedTimeOne;
        std::chrono::duration<double> elapsedTimeTwo;

        descriptorType descriptorOne = generateDescriptorsForObject(meshOne, 3, hardware.value(), elapsedTimeOne);
        descriptorType descriptorTwo = generateDescriptorsForObject(meshTwo, 3, hardware.value(), elapsedTimeTwo);

        std::cout << "d One length: " << std::get<3>(descriptorOne).length << std::endl;
        std::cout << "Mesh One vertex count: " << meshOne.vertexCount << std::endl;

        double similarity = calculateSimilarity<ShapeDescriptor::ShapeContextDescriptor>(std::get<3>(descriptorOne), std::get<3>(descriptorTwo), metadata, 1, true);

        std::cout << "Similarity: " << similarity << std::endl;

        // std::cout << "Time (descriptor one): " << elapsedTimeOne.count() << std::endl;
        // std::cout << "Time (descriptor two): " << elapsedTimeTwo.count() << std::endl;

        ShapeDescriptor::free::mesh(meshOne);
        ShapeDescriptor::free::mesh(meshTwo);
    }
    else if (objectsFolder.value() != "" && originalsFolderName.value() != "" && (originalObject.value() == "" && comparisonObject.value() == ""))
    {
        runDate = std::chrono::steady_clock::now().time_since_epoch().count();
        std::cout << "Comparing all objects in folder..." << std::endl;
        multipleObjectsBenchmark(objectsFolder.value(), originalsFolderName.value(), outputPath.value(), hardware.value(), compareFolder.value());

        std::string originalObjectsDataPath = outputPath.value() + std::to_string(runDate) + "/" + originalsFolderName.value() + ".json";

        std::ofstream outFile(originalObjectsDataPath);
        outFile << originalObjectsData.dump(4);
        outFile.close();
    }
    else
    {
        std::cout << "Wrong inputs, exiting..." << std::endl;
    }

    return 0;
}