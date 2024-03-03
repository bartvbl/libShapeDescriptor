#pragma once

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "helper_math.h"
#include "helper_cuda.h"
#define CUDA_REGION(contents) contents
#else
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#define CUDA_REGION(contents) throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif

#include <shapeDescriptor/libraryBuildSettings.h>
#include <shapeDescriptor/geometryTypes.h>
#include <shapeDescriptor/containerTypes.h>
#include <filesystem>
#include <cassert>
#include <vector>
#include <array>

#ifndef __SHAPE_DESCRIPTOR_HEADER_INCLUDED
#define __SHAPE_DESCRIPTOR_HEADER_INCLUDED
#endif


namespace ShapeDescriptor {

    // -- Descriptor types --
    // Various sizes are hardcoded and can be changed in libraryBuildSettings.h

    struct SpinImageDescriptor {
        spinImagePixelType contents[spinImageWidthPixels * spinImageWidthPixels];
    };

    struct RICIDescriptor {
        radialIntersectionCountImagePixelType contents[spinImageWidthPixels * spinImageWidthPixels];
    };

    struct QUICCIDescriptor {
        unsigned int contents[(spinImageWidthPixels * spinImageWidthPixels) / (sizeof(uint32_t) * 8)];
    };

    struct FPFHDescriptor {
        float contents[3 * FPFH_BINS_PER_FEATURE];
    };

    struct ShapeContextDescriptor {
        shapeContextBinType contents[
                SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT *
                SHAPE_CONTEXT_VERTICAL_SLICE_COUNT *
                SHAPE_CONTEXT_LAYER_COUNT];
    };



    // -- Library Types --

    struct OrientedPoint {
        ShapeDescriptor::cpu::float3 vertex;
        ShapeDescriptor::cpu::float3 normal;

        bool operator==(ShapeDescriptor::OrientedPoint other) const {
            return vertex == other.vertex && normal == other.normal;
        }
    };

    inline std::ostream & operator<<(std::ostream &os, const ShapeDescriptor::OrientedPoint point) {
        os << "OrientedPoint (vertex: " << &point.vertex <<  ", normal: " << &point.normal << ")";
        return os;
    }
    namespace cpu {
        struct BoundingBox {
            ShapeDescriptor::cpu::float3 min;
            ShapeDescriptor::cpu::float3 max;
        };
    }
    namespace gpu {
        struct BoundingBox {
            float3 min;
            float3 max;
        };
    }



    struct SampleBounds {
        size_t sampleCount;
        float areaStart;
        float areaEnd;
        size_t sampleStartIndex;
    };

    struct QUICCIDescriptorFileHeader {
        std::array<char, 5> fileID;
        size_t imageCount;
        unsigned int descriptorWidthPixels;
    };

    enum class RecomputeNormals {
        DO_NOT_RECOMPUTE, ALWAYS_RECOMPUTE, RECOMPUTE_IF_MISSING
    };

    template <typename ScoreType>
    struct SearchResults {
        size_t indices[SEARCH_RESULT_COUNT];
        ScoreType scores[SEARCH_RESULT_COUNT];
    };



    // -- Execution times --

    struct QUICCIExecutionTimes {
        double generationTimeSeconds = 0;
        double meshScaleTimeSeconds = 0;
        double redistributionTimeSeconds = 0;
        double totalExecutionTimeSeconds = 0;
    };
    struct RICIExecutionTimes {
        double generationTimeSeconds = 0;
        double meshScaleTimeSeconds = 0;
        double redistributionTimeSeconds = 0;
        double totalExecutionTimeSeconds = 0;
    };
    struct SIExecutionTimes {
        double totalExecutionTimeSeconds = 0;
        double initialisationTimeSeconds = 0;
        double generationTimeSeconds = 0;
    };
    struct SCExecutionTimes {
        double totalExecutionTimeSeconds = 0;
        double initialisationTimeSeconds = 0;
        double generationTimeSeconds = 0;
        double pointCountingTimeSeconds = 0;
    };
    struct FPFHExecutionTimes {
        double totalExecutionTimeSeconds = 0;
        double originReformatExecutionTimeSeconds = 0;
        double originSPFHGenerationExecutionTimeSeconds = 0;
        double pointCloudSPFHGenerationExecutionTimeSeconds = 0;
        double fpfhGenerationExecutionTimeSeconds = 0;
    };


    // -- Functions for generating descriptors using various methods --
    // The algorithm will run where the data is provided. That is, if data resides on the GPU, the algorithm
    // is used that runs on the GPU.

    cpu::array<QUICCIDescriptor> generateQUICCImages(
            cpu::Mesh device_mesh,
            cpu::array<OrientedPoint> device_descriptorOrigins,
            float supportRadius,
            QUICCIExecutionTimes* executionTimes = nullptr);

    cpu::array<QUICCIDescriptor> generatePartialityResistantQUICCImages(
            cpu::Mesh device_mesh,
            cpu::array<OrientedPoint> device_descriptorOrigins,
            float supportRadius,
            QUICCIExecutionTimes* executionTimes = nullptr);

    gpu::array<QUICCIDescriptor> generateQUICCImages(
            gpu::Mesh device_mesh,
            gpu::array<OrientedPoint> device_descriptorOrigins,
            float supportRadius,
            QUICCIExecutionTimes* executionTimes = nullptr);

    gpu::array<QUICCIDescriptor> generatePartialityResistantQUICCImages(
            gpu::Mesh device_mesh,
            gpu::array<OrientedPoint> device_descriptorOrigins,
            float supportRadius,
            QUICCIExecutionTimes* executionTimes = nullptr);

    cpu::array<RICIDescriptor> generateRadialIntersectionCountImages(
            cpu::Mesh mesh,
            cpu::array<OrientedPoint> descriptorOrigins,
            float supportRadius,
            RICIExecutionTimes* executionTimes = nullptr);

    gpu::array<RICIDescriptor> generateRadialIntersectionCountImages(
            gpu::Mesh device_mesh,
            gpu::array<OrientedPoint> device_descriptorOrigins,
            float supportRadius,
            RICIExecutionTimes* executionTimes = nullptr);

    cpu::array<SpinImageDescriptor> generateSpinImages(
            cpu::PointCloud pointCloud,
            cpu::array<OrientedPoint> descriptorOrigins,
            float supportRadius,
            float supportAngleDegrees,
            SIExecutionTimes* executionTimes = nullptr);

    gpu::array<SpinImageDescriptor> generateSpinImages(
            gpu::PointCloud device_pointCloud,
            gpu::array<OrientedPoint> device_descriptorOrigins,
            float supportRadius,
            float supportAngleDegrees,
            SIExecutionTimes* executionTimes = nullptr);

    cpu::array<ShapeDescriptor::ShapeContextDescriptor> generate3DSCDescriptors(
            cpu::PointCloud pointCloud,
            cpu::array<OrientedPoint> imageOrigins,
            float pointDensityRadius,
            float minSupportRadius,
            float maxSupportRadius,
            ShapeDescriptor::SCExecutionTimes* executionTimes = nullptr);

    gpu::array<ShapeContextDescriptor> generate3DSCDescriptors(
            gpu::PointCloud device_pointCloud,
            gpu::array<OrientedPoint> device_descriptorOrigins,
            float pointDensityRadius,
            float minSupportRadius,
            float maxSupportRadius,
            SCExecutionTimes* executionTimes = nullptr);

    gpu::array<FPFHDescriptor> generateFPFHHistograms(
            gpu::PointCloud device_pointCloud,
            gpu::array<OrientedPoint> device_descriptorOrigins,
            float supportRadius,
            FPFHExecutionTimes* executionTimes = nullptr);


    // -- Execution times structs for search methods --

    struct SCSearchExecutionTimes {
        double totalExecutionTimeSeconds = 0;
        double searchExecutionTimeSeconds = 0;
    };
    struct FPFHSearchExecutionTimes {
        double totalExecutionTimeSeconds = 0;
        double searchExecutionTimeSeconds = 0;
    };
    struct QUICCISearchExecutionTimes {
        double totalExecutionTimeSeconds = 0;
        double searchExecutionTimeSeconds = 0;
    };
    struct RICISearchExecutionTimes {
        double totalExecutionTimeSeconds = 0;
        double searchExecutionTimeSeconds = 0;
    };
    struct SISearchExecutionTimes {
        double totalExecutionTimeSeconds = 0;
        double searchExecutionTimeSeconds = 0;
        double averagingExecutionTimeSeconds = 0;
    };

    struct QUICCIDistances {
        unsigned int clutterResistantDistance = 0;
        unsigned int hammingDistance = 0;
        float weightedHammingDistance = 0;
        unsigned int needleImageBitCount = 0;
    };

#if QUICCI_DISTANCE_FUNCTION == WEIGHTED_HAMMING_DISTANCE
    typedef float quicciDistanceType;
#else
    typedef uint32_t quicciDistanceType;
#endif


    // -- Rank computation methods --
    // These take in a list of needle and haystack descriptors
    // For each needle descriptor, the index is computed where that descriptor would end up in a hypothetical list of search results.
    // Index 0 means the top of the list.

    cpu::array<unsigned int> compute3DSCSearchResultRanks(
            gpu::array<ShapeContextDescriptor> device_needleDescriptors,
            size_t needleDescriptorSampleCount,
            gpu::array<ShapeContextDescriptor> device_haystackDescriptors,
            size_t haystackDescriptorSampleCount,
            SCSearchExecutionTimes* executionTimes = nullptr);

    cpu::array<unsigned int> computeFPFHSearchResultRanks(
            gpu::array<FPFHDescriptor> device_needleDescriptors,
            gpu::array<FPFHDescriptor> device_haystackDescriptors,
            FPFHSearchExecutionTimes* executionTimes = nullptr);

    cpu::array<unsigned int> computeQUICCImageSearchResultRanks(
            gpu::array<QUICCIDescriptor> device_needleDescriptors,
            gpu::array<QUICCIDescriptor> device_haystackDescriptors,
            QUICCISearchExecutionTimes* executionTimes = nullptr);

    cpu::array<unsigned int> computeRadialIntersectionCountImageSearchResultRanks(
            gpu::array<RICIDescriptor> device_needleDescriptors,
            gpu::array<RICIDescriptor> device_haystackDescriptors,
            RICISearchExecutionTimes* executionTimes = nullptr);

    cpu::array<unsigned int> computeSpinImageSearchResultRanks(
            gpu::array<SpinImageDescriptor> device_needleDescriptors,
            gpu::array<SpinImageDescriptor> device_haystackDescriptors,
            SISearchExecutionTimes* executionTimes = nullptr);


    // -- Compute 1 to 1 descriptor distances using a given distance function --

    cpu::array<float> compute3DSCElementWiseSquaredDistances(
            gpu::array<ShapeContextDescriptor> device_descriptors,
            size_t descriptorSampleCount,
            gpu::array<ShapeContextDescriptor> device_correspondingDescriptors,
            size_t correspondingDescriptorsSampleCount);

    cpu::array<float> computeFPFHElementWiseEuclideanDistances(
            gpu::array<FPFHDescriptor> device_descriptors,
            gpu::array<FPFHDescriptor> device_correspondingDescriptors);

    cpu::array<QUICCIDistances> computeQUICCIElementWiseDistances(
            gpu::array<QUICCIDescriptor> device_descriptors,
            gpu::array<QUICCIDescriptor> device_correspondingDescriptors);

    cpu::array<float> computeQUICCIElementWiseWeightedHammingDistances(
            gpu::array<QUICCIDescriptor> device_descriptors,
            gpu::array<QUICCIDescriptor> device_correspondingDescriptors);

    cpu::array<int> computeRICIElementWiseModifiedSquareSumDistances(
            gpu::array<RICIDescriptor> device_descriptors,
            gpu::array<RICIDescriptor> device_correspondingDescriptors);

    cpu::array<float> computeSIElementWiseEuclideanDistances(
            gpu::array<SpinImageDescriptor> device_descriptors,
            gpu::array<SpinImageDescriptor> device_correspondingDescriptors);

    cpu::array<float> computeSIElementWisePearsonCorrelations(
            gpu::array<SpinImageDescriptor> device_descriptors,
            gpu::array<SpinImageDescriptor> device_correspondingDescriptors);


    // -- Direct search functions, returns a list of best matching descriptors for each needle descriptor --

    cpu::array<SearchResults<quicciDistanceType>> findQUICCImagesInHaystack(
            gpu::array<QUICCIDescriptor> device_needleDescriptors,
            gpu::array<QUICCIDescriptor> device_haystackDescriptors);

    cpu::array<SearchResults<unsigned int>> findRadialIntersectionCountImagesInHaystack(
            gpu::array<RICIDescriptor> device_needleDescriptors,
            gpu::array<RICIDescriptor> device_haystackDescriptors);

    cpu::array<SearchResults<float>> findSpinImagesInHaystack(
            gpu::array<SpinImageDescriptor> device_needleDescriptors,
            gpu::array<SpinImageDescriptor> device_haystackDescriptors);





    // -- Utility functions --

    cpu::Mesh copyToCPU(gpu::Mesh deviceMesh);
    cpu::PointCloud copyToCPU(gpu::PointCloud deviceMesh);
    cpu::array<cpu::float3> copyToCPU(gpu::VertexList vertexList);

    gpu::Mesh copyToGPU(cpu::Mesh hostMesh);
    gpu::PointCloud copyToGPU(cpu::PointCloud hostMesh);
    gpu::VertexList copyToGPU(cpu::array<cpu::float3> hostArray);

    template<typename T>
    void free(cpu::array<T> &arrayToFree) {
        delete[] arrayToFree.content;
        arrayToFree.content = nullptr;
    }

    template<typename T>
    void free(gpu::array<T> &arrayToFree) {
        CUDA_REGION(
        cudaFree(arrayToFree.content);
        arrayToFree.content = nullptr;
        )
    }
    void free(cpu::Mesh &meshToFree);
    void free(gpu::Mesh &meshToFree);
    void free(cpu::PointCloud &cloudToFree);
    void free(gpu::PointCloud &cloudToFree);



    gpu::BoundingBox computeBoundingBox(gpu::PointCloud device_pointCloud);
    cpu::BoundingBox computeBoundingBox(cpu::PointCloud pointCloud);
    double calculateMeshSurfaceArea(const cpu::Mesh& mesh);
    gpu::array<unsigned int> computePointDensities(float pointDensityRadius, gpu::PointCloud device_pointCloud);
    cpu::array<unsigned int> computePointDensities(float pointDensityRadius, cpu::PointCloud pointCloud);
    size_t compressBytes(void* outputBuffer, size_t outputBufferCapacity,
                         const void* inputBuffer, size_t inputBufferSize);
    size_t compressBytesMultithreaded(void* outputBuffer, size_t outputBufferCapacity,
                                      const void* inputBuffer, size_t inputBufferSize,
                                      unsigned numThreads);
    size_t decompressBytes(void* outputBuffer, size_t outputBufferCapacity,
                           const void* inputBuffer, size_t inputBufferCapacity);
    size_t decompressBytesMultithreaded(void* outputBuffer, size_t outputBufferCapacity,
                                        const void* inputBuffer, size_t inputBufferCapacity,
                                        unsigned int numThreads);
    size_t computeMaxCompressedBufferSize(size_t inputBufferSize);


    /* File reading */
    cpu::Mesh loadMeshFromCompressedGeometryFile(const std::filesystem::path &filePath);
    cpu::PointCloud readPointCloudFromCompressedGeometryFile(const std::filesystem::path &filePath);
    cpu::array<QUICCIDescriptor> readCompressedQUICCIDescriptors(const std::filesystem::path &dumpFileLocation, unsigned int decompressionThreadCount = 1);
    QUICCIDescriptorFileHeader readCompressedQUICCIDescriptorFileHeader(const std::filesystem::path &dumpFileLocation);

    cpu::Mesh loadMesh(std::filesystem::path src, RecomputeNormals recomputeNormals = RecomputeNormals::DO_NOT_RECOMPUTE);
    cpu::PointCloud loadPointCloud(std::filesystem::path src);

    cpu::Mesh loadGLTFMesh(std::filesystem::path, RecomputeNormals recomputeNormals = RecomputeNormals::DO_NOT_RECOMPUTE);
    cpu::Mesh loadOBJ(std::filesystem::path, RecomputeNormals recomputeNormals = RecomputeNormals::DO_NOT_RECOMPUTE);
    cpu::Mesh loadOFF(std::filesystem::path src);
    // Note: this is not a complete implementation of the file format.
    // Instead, it should capture the parts that are used most in practice.
    cpu::Mesh loadPLY(std::filesystem::path src, RecomputeNormals recomputeNormals = RecomputeNormals::DO_NOT_RECOMPUTE);

    cpu::PointCloud loadGLTFPointCloud(std::filesystem::path);
    cpu::PointCloud loadXYZ(std::filesystem::path src, bool readNormals = false, bool readColours = false);
    bool gltfContainsPointCloud(const std::filesystem::path& file);



    void writeCompressedGeometryFile(const ShapeDescriptor::cpu::Mesh &mesh, const std::filesystem::path &filePath, bool stripVertexColours = false);
    void writeCompressedGeometryFile(const ShapeDescriptor::cpu::PointCloud &cloud, const std::filesystem::path &filePath, bool stripVertexColours = false);

    void writeXYZ(std::filesystem::path destination, ShapeDescriptor::cpu::PointCloud pointCloud);

    void writeDescriptorImages(
            cpu::array<RICIDescriptor> hostDescriptors,
            std::filesystem::path imageDestinationFile,
            bool logarithmicImage = true,
            unsigned int imagesPerRow = 50,
            unsigned int imageLimit = 2000);

    void writeDescriptorImages(
            cpu::array<SpinImageDescriptor> hostDescriptors,
            std::filesystem::path imageDestinationFile,
            bool logarithmicImage = true,
            unsigned int imagesPerRow = 50,
            unsigned int imageLimit = 2000);

    void writeDescriptorImages(
            cpu::array<QUICCIDescriptor> hostDescriptors,
            std::filesystem::path imageDestinationFile,
            unsigned int imagesPerRow = 50,
            unsigned int imageLimit = 2000);

    void writeDescriptorImages(
            cpu::array<QUICCIDescriptor> hostDescriptors,
            std::filesystem::path imageDestinationFile,
            bool unusedOnlyExistsForCompatibility = false,
            unsigned int imagesPerRow = 50,
            unsigned int imageLimit = 2000);

    // Write an image where each channel shows a different descriptor.
    // Useful for comparing similarity of different QUICCI descriptors
    void writeDescriptorComparisonImage(
            std::filesystem::path imageDestinationFile,
            cpu::array<QUICCIDescriptor> blueChannelDescriptors = {0, nullptr},
            cpu::array<QUICCIDescriptor> greenChannelDescriptors = {0, nullptr},
            cpu::array<QUICCIDescriptor> redChannelDescriptors = {0, nullptr},
            unsigned int imagesPerRow = 50,
            unsigned int imageLimit = 2000);

    void writeOBJ(cpu::Mesh mesh, const std::filesystem::path outputFile);
    void writeOBJ(cpu::Mesh mesh, const std::filesystem::path outputFilePath,
              size_t highlightStartVertex, size_t highlightEndVertex);
    void writeOBJ(cpu::Mesh mesh, const std::filesystem::path &outputFilePath,
              cpu::array<float2> vertexTextureCoordinates, std::string textureMapPath);

    void writeCompressedQUICCIDescriptors(
            const std::filesystem::path &outputDumpFile,
            const cpu::array<QUICCIDescriptor> &images,
            unsigned int compressionThreadCount = 1);



    uint32_t hashMesh(const cpu::Mesh& mesh);
    uint32_t hashPointCloud(const cpu::PointCloud& cloud);
    bool compareMesh(const cpu::Mesh& mesh, const cpu::Mesh& otherMesh);
    bool comparePointCloud(const cpu::PointCloud& cloud, const cpu::PointCloud& otherCloud);

    cpu::Mesh scaleMesh(cpu::Mesh &model, cpu::Mesh &scaledModel, float spinImagePixelSize);

    cpu::Mesh fitMeshInsideSphereOfRadius(cpu::Mesh &input, float radius);

    void printQuicciDescriptor(QUICCIDescriptor &descriptor);

    bool isCUDASupportAvailable();
    // Returns the ID of the GPU on which the context was created
    int createCUDAContext(int forceGPU = -1);
    void printGPUProperties(unsigned int deviceIndex);

    std::vector<std::filesystem::path> listDirectory(const std::filesystem::path& directory);
    std::vector<std::filesystem::path> listDirectoryAndSubdirectories(const std::filesystem::path& directory);
    void writeCompressedFile(const char* buffer, size_t bufferSize, const std::filesystem::path &archiveFile, unsigned int threadCount = 1);
    std::vector<char> readCompressedFile(const std::filesystem::path &archiveFile, unsigned int threadCount = 1);
    std::vector<char> readCompressedFileUpToNBytes(const std::filesystem::path &archiveFile, size_t decompressedBytesToRead, unsigned int threadCount = 1);
    std::string generateUniqueFilenameString();

    cpu::array<OrientedPoint> generateSpinOriginBuffer(const cpu::Mesh &mesh);
    cpu::array<OrientedPoint> generateUniqueSpinOriginBuffer(const cpu::Mesh &mesh, std::vector<size_t>* indexMapping = nullptr);

    cpu::float3 computeTriangleNormal(cpu::float3 &triangleVertex0, cpu::float3 &triangleVertex1, cpu::float3 &triangleVertex2);

    namespace internal {
        float computeBinVolume(short verticalBinIndex, short layerIndex, float minSupportRadius, float maxSupportRadius);

        struct MeshSamplingBuffers {
            gpu::array<float> cumulativeAreaArray;
        };
    }

    cpu::PointCloud sampleMesh(cpu::Mesh mesh, size_t sampleCount, size_t randomSamplingSeed);


#ifdef __CUDACC__
    __inline__ __device__ unsigned int warpAllReduceSum(unsigned int val) {
        for (int mask = warpSize/2; mask > 0; mask /= 2)
            val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
        return val;
    }

    __inline__ __device__ float warpAllReduceSum(float val) {
        for (int mask = warpSize/2; mask > 0; mask /= 2)
            val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
        return val;
    }

    __inline__ __device__ float warpAllReduceMin(float val) {
        for (int mask = warpSize/2; mask > 0; mask /= 2)
            val = min(val, __shfl_xor_sync(0xFFFFFFFF, val, mask));
        return val;
    }
#endif
}

#include "compressedDescriptorIO.h"

// Allow inclusion into std::set
namespace std {
    template <> struct hash<ShapeDescriptor::OrientedPoint>
    {
        size_t operator()(const ShapeDescriptor::OrientedPoint& p) const
        {
            return std::hash<ShapeDescriptor::cpu::float3>()(p.vertex) ^ std::hash<ShapeDescriptor::cpu::float3>()(p.normal);
        }
    };
}