#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/read/MeshLoader.h>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <shapeDescriptor/cpu/quickIntersectionCountImageGenerator.h>
#include <shapeDescriptor/cpu/spinImageGenerator.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/gpu/spinImageSearcher.cuh>
#include <shapeDescriptor/common/types/methods/SpinImageDescriptor.h>
#include <benchmarking/utilities/descriptor/RICI.h>
#include <benchmarking/utilities/descriptor/QUICCI.h>
#include <benchmarking/utilities/descriptor/spinImage.h>
#include <benchmarking/utilities/descriptor/3dShapeContext.h>
#include <benchmarking/utilities/descriptor/FPFH.h>
#include <benchmarking/utilities/distance/similarity.h>
#include <benchmarking/utilities/metadata/generateFakeMetadata.h>
#include <benchmarking/utilities/metadata/prepareMetadata.h>
#include <benchmarking/utilities/metadata/transformDescriptor.h>
#include <tuple>
#include <iostream>
#include <fstream>
#include <arrrgh.hpp>
#include <vector>
#include <variant>
#include <map>
#include <json.hpp>
#include <ctime>
#include <chrono>
#include <git.h>
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

const auto runDate = std::chrono::system_clock::now();

std::string getRunDate()
{
    auto in_time_t = std::chrono::system_clock::to_time_t(runDate);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d-%H:%M:%S");
    return ss.str();
}

descriptorType generateDescriptorsForObject(ShapeDescriptor::cpu::Mesh mesh,
                                            int algorithm,
                                            std::string hardware,
                                            std::chrono::duration<double> &elapsedTime,
                                            float supportRadius = 2.5f,
                                            float supportAngleDegrees = 60.0f,
                                            float pointDensityRadius = 0.2f,
                                            float minSupportRadius = 0.1f,
                                            float maxSupportRadius = 2.5f,
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
            mesh, hardware, supportRadius, supportAngleDegrees, pointCloudSampleCount, randomSeed, elapsedTime);
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
float calculateAverageSimilartyFromDistancesArray(ShapeDescriptor::cpu::array<T> distances)
{
    float simSum = 0;

    for (int i = 0; i < distances.length; i++)
    {
        float sim = 1 / (1 + distances[i]);
        if (!isnan(sim))
            simSum += sim;
    }
    float avgSim = simSum / distances.length;

    return avgSim;
}

template <typename T>
float calculateAverageSimilarity(ShapeDescriptor::cpu::array<T> distances)
{
    float simSum = 0;

    for (int i = 0; i < distances.length; i++)
    {
        simSum += isnan(distances[i]) ? 0 : distances[i];
    }

    float avgSim = simSum / distances.length;

    return avgSim;
}

template <typename T>
double calculateSimilarity(ShapeDescriptor::cpu::array<T> dOriginal, ShapeDescriptor::cpu::array<T> dComparison, int distanceFunction, bool freeArray)
{
    double sim = Benchmarking::utilities::distance::similarityBetweenTwoDescriptors<T>(dOriginal, dComparison, distanceFunction);

    if (freeArray)
    {
        ShapeDescriptor::free::array(dOriginal);
        ShapeDescriptor::free::array(dComparison);
    }

    return sim;
}

template <typename T>
void freeDescriptorType(ShapeDescriptor::cpu::array<T> descriptor)
{
    ShapeDescriptor::free::array(descriptor);
}

int getNumberOfFilesInFolder(std::string folderPath)
{
    int numberOfFiles = 0;
    for (const auto &entry : std::filesystem::directory_iterator(folderPath))
    {
        numberOfFiles++;
    }

    return numberOfFiles;
}

void multipleObjectsBenchmark(std::string objectsFolder, std::string originalsFolderName, std::string jsonPath, std::string hardware, std::string compareFolder, std::string previousRunPath)
{
    std::vector<std::string> folders;
    std::string originalObjectFolderPath;

    json previousRun;
    if (std::filesystem::exists(previousRunPath))
    {
        std::ifstream jsonFile(previousRunPath);
        previousRun = json::parse(jsonFile);
    }

    // This is hard coded for now, as this fits how we have structured the folder. Should be edited if you want the code more dynamic:^)
    std::string originalObjectCategory = "0-100";

    std::string outputDirectory = jsonPath + "/" + getRunDate();

    float supportRadius = 1.5f;
    float supportAngleDegrees = 0.0f;
    float pointDensityRadius = 0.2f;
    float minSupportRadius = 0.1f;
    float maxSupportRadius = 2.5f;
    size_t pointCloudSampleCount = 200000;
    size_t randomSeed = 4917133789385064;

    for (auto &p : std::filesystem::directory_iterator(objectsFolder))
    {
        if (p.is_directory())
        {
            if (originalObjectFolderPath.empty() && p.path().string().substr(p.path().string().find_last_of("/") + 1) == originalsFolderName)
            {
                originalObjectFolderPath = p.path().string() + "/" + originalObjectCategory;
            }
            else if (compareFolder == "")
            {
                folders.push_back(p.path().string());
            }
            else if (p.path().string().substr(p.path().string().find_last_of("/") + 1) == compareFolder)
            {
                folders.push_back(p.path().string());
            }
        }
    }

    for (std::string folder : folders)
    {
        std::chrono::steady_clock::time_point timeStart = std::chrono::steady_clock::now();
        json jsonOutput;
        std::string comparisonFolderName = folder.substr(folder.find_last_of("/") + 1);

        if (!std::filesystem::exists(outputDirectory + "-" + comparisonFolderName))
        {
            std::filesystem::create_directory(outputDirectory + "-" + comparisonFolderName);
        }

        jsonOutput["runDate"] = getRunDate();
        jsonOutput["hardware"]["type"] = hardware;

        jsonOutput["buildInfo"]["commit"] = GitMetadata::CommitSHA1();
        jsonOutput["buildInfo"]["commit_author"] = GitMetadata::AuthorEmail();
        jsonOutput["buildInfo"]["commit_date"] = GitMetadata::CommitSubject();

        jsonOutput["static"] = {};
        jsonOutput["static"]["supportRadius"] = supportRadius;
        jsonOutput["static"]["supportAngleDegrees"] = supportAngleDegrees;
        jsonOutput["static"]["pointDensityRadius"] = pointDensityRadius;
        jsonOutput["static"]["minSupportRadius"] = minSupportRadius;
        jsonOutput["static"]["maxSupportRadius"] = maxSupportRadius;
        jsonOutput["static"]["pointCloudSampleCount"] = pointCloudSampleCount;
        jsonOutput["static"]["randomSeed"] = randomSeed;

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
        jsonOutput["hardware"]["gpu"]["name"] = GPUInfo.name;
        jsonOutput["hardware"]["gpu"]["clockRate"] = GPUInfo.clockRate;
        jsonOutput["hardware"]["gpu"]["memory"] = GPUInfo.memory;
#endif

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

                std::cout << "Comparing object " << fileName << " in category " << category << std::endl;

                try
                {
                    originalObjectPath = originalFolder + "/" + fileName + ".obj";
                    comparisonObjectPath = comparisonFolder + "/" + fileName + ".obj";

                    meshOriginal = ShapeDescriptor::utilities::loadMesh(originalObjectPath);
                    meshComparison = ShapeDescriptor::utilities::loadMesh(comparisonObjectPath);

                    metadata = Benchmarking::utilities::metadata::prepareMetadata(comparisonFolder + "/" + fileName + ".txt", meshOriginal.vertexCount);
                }
                catch (const std::exception e)
                {
                    std::cout << "Comparison object " << comparisonObjectPath << " not found..." << std::endl;
                    continue;
                }

                jsonOutput["results"][fileName][comparisonFolderName]["vertexCounts"][category]["vertexCount"] = meshComparison.vertexCount;

                for (auto a : descriptorAlgorithms)
                {
                    std::chrono::duration<double> elapsedSecondsDescriptorComparison;
                    std::chrono::duration<double> elapsedSecondsDescriptorOriginal;

                    if (previousRun["results"][fileName][comparisonFolderName][a.second][category].size() > 0)
                    {
                        jsonOutput["results"][fileName][comparisonFolderName][a.second][category] = previousRun["results"][fileName][comparisonFolderName][a.second][category];
                        continue;
                    }

                    descriptorType originalObject = generateDescriptorsForObject(
                        meshOriginal, a.first, hardware, elapsedSecondsDescriptorOriginal,
                        supportRadius, supportAngleDegrees, pointDensityRadius, minSupportRadius, maxSupportRadius,
                        pointCloudSampleCount, randomSeed);

                    descriptorType comparisonObject = generateDescriptorsForObject(
                        meshComparison, a.first, hardware, elapsedSecondsDescriptorComparison,
                        supportRadius, supportAngleDegrees, pointDensityRadius, minSupportRadius, maxSupportRadius,
                        pointCloudSampleCount, randomSeed);

                    descriptorType transformedOriginalObject;
                    descriptorType transformedComparisonObject;

                    switch (a.first)
                    {
                    case 0:
                    {
                        std::vector<ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>> transformed =
                            Benchmarking::utilities::metadata::transformDescriptorsToMatchMetadata(std::get<0>(originalObject), std::get<0>(comparisonObject), metadata);

                        transformedOriginalObject = transformed.at(0);
                        transformedComparisonObject = transformed.at(1);

                        freeDescriptorType(std::get<0>(originalObject));
                        freeDescriptorType(std::get<0>(comparisonObject));
                        break;
                    }
                    case 1:
                    {
                        std::vector<ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor>> transformed =
                            Benchmarking::utilities::metadata::transformDescriptorsToMatchMetadata(std::get<1>(originalObject), std::get<1>(comparisonObject), metadata);

                        transformedOriginalObject = transformed.at(0);
                        transformedComparisonObject = transformed.at(1);

                        freeDescriptorType(std::get<1>(originalObject));
                        freeDescriptorType(std::get<1>(comparisonObject));
                        break;
                    }
                    case 2:
                    {
                        std::vector<ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor>> transformed =
                            Benchmarking::utilities::metadata::transformDescriptorsToMatchMetadata(std::get<2>(originalObject), std::get<2>(comparisonObject), metadata);

                        transformedOriginalObject = transformed.at(0);
                        transformedComparisonObject = transformed.at(1);

                        freeDescriptorType(std::get<2>(originalObject));
                        freeDescriptorType(std::get<2>(comparisonObject));
                        break;
                    }
                    case 3:
                    {
                        std::vector<ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor>> transformed =
                            Benchmarking::utilities::metadata::transformDescriptorsToMatchMetadata(std::get<3>(originalObject), std::get<3>(comparisonObject), metadata);

                        transformedOriginalObject = transformed.at(0);
                        transformedComparisonObject = transformed.at(1);

                        freeDescriptorType(std::get<3>(originalObject));
                        freeDescriptorType(std::get<3>(comparisonObject));
                        break;
                    }
                    case 4:
                    {
                        std::vector<ShapeDescriptor::cpu::array<ShapeDescriptor::FPFHDescriptor>> transformed =
                            Benchmarking::utilities::metadata::transformDescriptorsToMatchMetadata(std::get<4>(originalObject), std::get<4>(comparisonObject), metadata);

                        transformedOriginalObject = transformed.at(0);
                        transformedComparisonObject = transformed.at(1);

                        freeDescriptorType(std::get<4>(originalObject));
                        freeDescriptorType(std::get<4>(comparisonObject));
                        break;
                    }
                    default:
                    {
                        std::vector<ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>> transformed =
                            Benchmarking::utilities::metadata::transformDescriptorsToMatchMetadata(std::get<0>(originalObject), std::get<0>(comparisonObject), metadata);

                        transformedOriginalObject = transformed.at(0);
                        transformedComparisonObject = transformed.at(1);

                        freeDescriptorType(std::get<0>(originalObject));
                        freeDescriptorType(std::get<0>(comparisonObject));
                        break;
                    }
                    }

                    for (auto d : distanceFunctions)
                    {
                        bool freeArray = d.first == distanceFunctions.size() - 1;
                        double sim = 0;

                        std::chrono::steady_clock::time_point distanceTimeStart;
                        std::chrono::steady_clock::time_point distanceTimeEnd;

                        switch (a.first)
                        {
                        case 0:
                        {
                            ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> original =
                                std::get<0>(transformedOriginalObject);

                            ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> comparison =
                                std::get<0>(transformedComparisonObject);

                            if (originalObjectsData["results"].find(fileName) == originalObjectsData["results"].end())
                            {
                                originalObjectsData["results"][fileName][a.second]["generationTime"] = elapsedSecondsDescriptorOriginal.count();
                                // The length of the descriptor is the exact number of verticies
                                // While the vertex count in the mesh class is just faces * 3, which is can sometimes be not accurate
                                originalObjectsData["results"][fileName]["vertexCount"] = original.length;
                            }

                            distanceTimeStart = std::chrono::steady_clock::now();
                            sim = calculateSimilarity<ShapeDescriptor::RICIDescriptor>(original, comparison, d.first, freeArray);
                            distanceTimeEnd = std::chrono::steady_clock::now();
                            break;
                        }
                        case 1:
                        {
                            ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> original =
                                std::get<1>(transformedOriginalObject);

                            ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> comparison =
                                std::get<1>(transformedComparisonObject);

                            if (originalObjectsData["results"].find(fileName) == originalObjectsData["results"].end())
                            {
                                originalObjectsData["results"][fileName][a.second]["generationTime"] = elapsedSecondsDescriptorOriginal.count();
                                originalObjectsData["results"][fileName]["vertexCount"] = original.length;
                            }

                            distanceTimeStart = std::chrono::steady_clock::now();
                            sim = calculateSimilarity<ShapeDescriptor::QUICCIDescriptor>(original, comparison, d.first, freeArray);
                            distanceTimeEnd = std::chrono::steady_clock::now();
                            break;
                        }
                        case 2:
                        {
                            ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> original =
                                std::get<2>(transformedOriginalObject);

                            ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> comparison =
                                std::get<2>(transformedComparisonObject);

                            if (originalObjectsData["results"].find(fileName) == originalObjectsData["results"].end())
                            {
                                originalObjectsData["results"][fileName][a.second]["generationTime"] = elapsedSecondsDescriptorOriginal.count();
                                originalObjectsData["results"][fileName]["vertexCount"] = original.length;
                            }

                            distanceTimeStart = std::chrono::steady_clock::now();
                            sim = calculateSimilarity<ShapeDescriptor::SpinImageDescriptor>(original, comparison, d.first, freeArray);
                            distanceTimeEnd = std::chrono::steady_clock::now();
                            break;
                        }
                        case 3:
                        {
                            ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> original =
                                std::get<3>(transformedOriginalObject);

                            ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> comparison =
                                std::get<3>(transformedComparisonObject);

                            if (originalObjectsData["results"].find(fileName) == originalObjectsData["results"].end())
                            {
                                originalObjectsData["results"][fileName][a.second]["generationTime"] = elapsedSecondsDescriptorOriginal.count();
                                originalObjectsData["results"][fileName]["vertexCount"] = original.length;
                            }

                            distanceTimeStart = std::chrono::steady_clock::now();
                            sim = calculateSimilarity<ShapeDescriptor::ShapeContextDescriptor>(original, comparison, d.first, freeArray);
                            distanceTimeEnd = std::chrono::steady_clock::now();
                            break;
                        }
                        case 4:
                        {
                            ShapeDescriptor::cpu::array<ShapeDescriptor::FPFHDescriptor> original =
                                std::get<4>(transformedOriginalObject);

                            ShapeDescriptor::cpu::array<ShapeDescriptor::FPFHDescriptor> comparison =
                                std::get<4>(transformedComparisonObject);

                            if (originalObjectsData["results"].find(fileName) == originalObjectsData["results"].end())
                            {
                                originalObjectsData["results"][fileName][a.second]["generationTime"] = elapsedSecondsDescriptorOriginal.count();
                                originalObjectsData["results"][fileName]["vertexCount"] = original.length;
                            }
                            distanceTimeStart = std::chrono::steady_clock::now();
                            sim = calculateSimilarity<ShapeDescriptor::FPFHDescriptor>(original, comparison, d.first, freeArray);
                            distanceTimeEnd = std::chrono::steady_clock::now();
                            break;
                        }
                        default:
                        {
                            ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> original =
                                std::get<0>(transformedOriginalObject);

                            ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> comparison =
                                std::get<0>(transformedComparisonObject);

                            if (originalObjectsData["results"].find(fileName) == originalObjectsData["results"].end())
                            {
                                originalObjectsData["results"][fileName][a.second]["generationTime"] = elapsedSecondsDescriptorOriginal.count();
                                originalObjectsData["results"][fileName]["vertexCount"] = original.length;
                            }

                            distanceTimeStart = std::chrono::steady_clock::now();
                            sim = calculateSimilarity<ShapeDescriptor::RICIDescriptor>(original, comparison, d.first, freeArray);
                            distanceTimeEnd = std::chrono::steady_clock::now();
                            break;
                        }
                        }

                        std::chrono::duration<double> elapsedSecondsDistance = distanceTimeEnd - distanceTimeStart;

                        jsonOutput["results"][fileName][comparisonFolderName][a.second][category]["generationTime"] = elapsedSecondsDescriptorComparison.count();
                        jsonOutput["results"][fileName][comparisonFolderName][a.second][category][d.second]["similarity"] = (double)sim;
                        jsonOutput["results"][fileName][comparisonFolderName][a.second][category][d.second]["time"] = elapsedSecondsDistance.count();
                    }
                }
                ShapeDescriptor::free::mesh(meshOriginal);
                ShapeDescriptor::free::mesh(meshComparison);
                metadata.clear();

                std::chrono::steady_clock::time_point timeAfter = std::chrono::steady_clock::now();
                std::chrono::duration<double> currentTotalRunTime = timeAfter - timeStart;

                jsonOutput["runTime"] = currentTotalRunTime.count();

                std::string outputFilePath = outputDirectory + "-" + comparisonFolderName + "/" + comparisonFolderName + ".json";
                std::ofstream outFile(outputFilePath);
                outFile << jsonOutput.dump(4);
                outFile.close();

                std::cout << "Results stored to " << outputFilePath << std::endl;
            }
        }
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
    const auto &previousRunFile = parser.add<std::string>("previous-run", "Path to a JSON file containing data from a previous run", 'P', arrrgh::Optional, "");
    const auto &metadataPath = parser.add<std::string>("metadata", "Path to metadata describing which vertecies that are changed", 'm', arrrgh::Optional, "");
    const auto &outputPath = parser.add<std::string>("output-path", "Path to the output", 'p', arrrgh::Optional, "");
    const auto &descriptorAlgorithm = parser.add<int>("descriptor-algorithm", "Which descriptor algorithm to use [0 for radial-intersection-count-images, 1 for quick-intersection-count-change-images ...will add more:)]", 'a', arrrgh::Optional, 0);
    const auto &distanceAlgorithm = parser.add<int>("distance-algorithm", "Which distance algorithm to use [0 for euclidian, ...will add more:)]", 'd', arrrgh::Optional, 0);
    const auto &hardware = parser.add<std::string>("hardware-type", "cpu or gpu (gpu is default, as cpu doesn't support all the descriptors)", 't', arrrgh::Optional, "gpu");
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

    if (originalObject.value() != "")
    {
        json spinImageTest;

        int numberOfObjects = 1000;
        std::string originalObjectsPath = "/mnt/VOID/projects/shape_descriptors_benchmark/Dataset/NewRecalculatedNormals/0-100/";
        std::string comparisonObjectsPath = "/mnt/VOID/projects/shape_descriptors_benchmark/Dataset/OverlappingObjects/15.1-25.0/";

        std::string outPath = "/mnt/VOID/projects/shape_descriptors_benchmark/Output/spinImageDistancetest";
        std::filesystem::create_directory(outPath);

        std::string output = "dataset,object,descriptor,category,distanceFunction,similarity,time\n";

        for (int objectNumber = 0; objectNumber < numberOfObjects; objectNumber++)
        {
            std::cout << "Testing object " << objectNumber << std::endl;

            std::string objectStr = std::to_string(objectNumber);
            std::string objectName = std::string(4 - objectStr.length(), '0') + objectStr;

            std::string metadataFile = comparisonObjectsPath + objectName + "/" + objectName + ".txt";
            std::vector<std::variant<int, std::string>> metadata = Benchmarking::utilities::metadata::prepareMetadata(metadataFile);

            std::filesystem::path objectOne = originalObjectsPath + objectName + "/" + objectName + ".obj";
            std::filesystem::path objectTwo = comparisonObjectsPath + objectName + "/" + objectName + ".obj";

            ShapeDescriptor::cpu::Mesh meshOne = ShapeDescriptor::utilities::loadMesh(objectOne);
            ShapeDescriptor::cpu::Mesh meshTwo = ShapeDescriptor::utilities::loadMesh(objectTwo);

            std::chrono::duration<double> elapsedTimeOne;
            std::chrono::duration<double> elapsedTimeTwo;

            ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> descriptorOne =
                std::get<2>(generateDescriptorsForObject(meshOne, 2, hardware.value(), elapsedTimeOne));
            ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> descriptorTwo =
                std::get<2>(generateDescriptorsForObject(meshTwo, 2, hardware.value(), elapsedTimeTwo));

            std::vector<ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor>>
                transformedDescriptors = Benchmarking::utilities::metadata::transformDescriptorsToMatchMetadata(descriptorOne, descriptorTwo, metadata);

            ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> transformedOriginal = transformedDescriptors.at(0);
            ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> transformedComparison = transformedDescriptors.at(1);

            ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> descriptorOneGPU = transformedOriginal.copyToGPU();
            ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> descriptorTwoGPU = transformedComparison.copyToGPU();

            std::chrono::steady_clock::time_point cosineTimeStart = std::chrono::steady_clock::now();
            ShapeDescriptor::cpu::array<float> similaritesCosine =
                ShapeDescriptor::gpu::computeSIElementWiseCosineSimilarity(descriptorOneGPU, descriptorTwoGPU);
            float cosineSim = calculateAverageSimilarity(similaritesCosine);
            std::chrono::steady_clock::time_point cosineTimeEnd = std::chrono::steady_clock::now();

            std::chrono::duration<double> cosineTime = cosineTimeEnd - cosineTimeStart;

            std::chrono::steady_clock::time_point pearsonTimeStart = std::chrono::steady_clock::now();
            ShapeDescriptor::cpu::array<float> distancesPearson =
                ShapeDescriptor::gpu::computeSIElementWisePearsonCorrelations(descriptorOneGPU, descriptorTwoGPU);
            float pearsonSim = calculateAverageSimilarity(distancesPearson);
            std::chrono::steady_clock::time_point pearsonTimeEnd = std::chrono::steady_clock::now();

            std::chrono::duration<double> pearsonTime = pearsonTimeEnd - pearsonTimeStart;

            std::cout << "Cosine similarity: " << cosineSim << std::endl;
            std::cout << "Pearson similarity: " << pearsonSim << std::endl;

            output += "OverlappingObjects," + objectName + ",SI,15.1-25.0,Cosine," + std::to_string(cosineSim) + "," + std::to_string(cosineTime.count()) + "\n";
            output += "OverlappingObjects," + objectName + ",SI,15.1-25.0,Pearson," + std::to_string(pearsonSim) + "," + std::to_string(pearsonTime.count()) + "\n";

            metadata.clear();

            ShapeDescriptor::free::array(similaritesCosine);
            ShapeDescriptor::free::array(distancesPearson);
            ShapeDescriptor::free::array(descriptorOne);
            ShapeDescriptor::free::array(descriptorTwo);
            ShapeDescriptor::free::array(transformedOriginal);
            ShapeDescriptor::free::array(transformedComparison);
            ShapeDescriptor::free::array(descriptorOneGPU);
            ShapeDescriptor::free::array(descriptorTwoGPU);
            ShapeDescriptor::free::mesh(meshOne);
            ShapeDescriptor::free::mesh(meshTwo);
        }

        std::ofstream outFile(outPath + "/spinImageDistanceTest.csv");
        outFile << output;
        outFile.close();
    }
    else if (objectsFolder.value() != "" && originalsFolderName.value() != "" && (originalObject.value() == "" && comparisonObject.value() == ""))
    {
        std::cout << "Comparing all objects in folder..." << std::endl;
        multipleObjectsBenchmark(objectsFolder.value(), originalsFolderName.value(), outputPath.value(), hardware.value(), compareFolder.value(), previousRunFile.value());

        std::string originalObjectsDataPath = outputPath.value() + "/" + getRunDate() + "-" + originalsFolderName.value() + "/" + originalsFolderName.value() + ".json";

        std::cout << "Writing original objects data to " << originalObjectsDataPath << std::endl;

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