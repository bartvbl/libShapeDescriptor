#include <shapeDescriptor/shapeDescriptor.h>




ShapeDescriptor::cpu::BoundingBox ShapeDescriptor::computeBoundingBox(ShapeDescriptor::cpu::PointCloud pointCloud) {
    ShapeDescriptor::cpu::BoundingBox boundingBox = {
            {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()},
            {-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()}};

    for(size_t vertexIndex = 0; vertexIndex < pointCloud.pointCount; vertexIndex++) {
        ShapeDescriptor::cpu::float3 vertex = pointCloud.vertices[vertexIndex];

        boundingBox.min = {
                std::min(boundingBox.min.x, vertex.x),
                std::min(boundingBox.min.y, vertex.y),
                std::min(boundingBox.min.z, vertex.z)
        };

        boundingBox.max = {
                std::max(boundingBox.max.x, vertex.x),
                std::max(boundingBox.max.y, vertex.y),
                std::max(boundingBox.max.z, vertex.z)
        };
    }
    return boundingBox;
}











inline unsigned int computeBinIndex(ShapeDescriptor::cpu::int3 binIndex, ShapeDescriptor::cpu::int3 binCounts) {
    return binIndex.z * binCounts.x * binCounts.y + binIndex.y * binCounts.x + binIndex.x;
}

inline uint32_t computeBinIndex(ShapeDescriptor::cpu::float3 vertex, ShapeDescriptor::cpu::BoundingBox boundingBox, double binSize, ShapeDescriptor::cpu::int3 binCounts) {
    ShapeDescriptor::cpu::float3 relativeToBoundingBox = vertex - boundingBox.min;

    ShapeDescriptor::cpu::int3 binIndex = {
            std::min(std::max(int(relativeToBoundingBox.x / binSize), 0), binCounts.x - 1),
            std::min(std::max(int(relativeToBoundingBox.y / binSize), 0), binCounts.y - 1),
            std::min(std::max(int(relativeToBoundingBox.z / binSize), 0), binCounts.z - 1)
    };

    uint32_t indexTableIndex = computeBinIndex(binIndex, binCounts);

    return indexTableIndex;
}

void countBinContents(
        ShapeDescriptor::cpu::PointCloud pointCloud,
        std::vector<uint32_t>& indexTable,
        ShapeDescriptor::cpu::BoundingBox boundingBox,
        ShapeDescriptor::cpu::int3 binCounts,
        double binSize) {

    for(uint32_t vertexIndex = 0; vertexIndex < pointCloud.pointCount; vertexIndex++) {
        ShapeDescriptor::cpu::float3 vertex = pointCloud.vertices[vertexIndex];

        uint32_t indexTableIndex = computeBinIndex(vertex, boundingBox, binSize, binCounts);

        indexTable.at(indexTableIndex)++;
    }
}

void countCumulativeBinIndices(std::vector<uint32_t>& indexTable, ShapeDescriptor::cpu::int3 binCounts, unsigned int pointCloudSize) {
    unsigned int cumulativeIndex = 0;
    for(int z = 0; z < binCounts.z; z++) {
        for(int y = 0; y < binCounts.y; y++) {
            for(int x = 0; x < binCounts.x; x++) {
                unsigned int binIndex = computeBinIndex({x, y, z}, binCounts);
                unsigned int binLength = indexTable.at(binIndex);
                indexTable.at(binIndex) = cumulativeIndex;
                cumulativeIndex += binLength;
            }
        }
    }
    assert(cumulativeIndex == pointCloudSize);
}

void computePointCounts(
        std::vector<uint32_t>& pointDensityArray,
        ShapeDescriptor::cpu::PointCloud pointCloud,
        ShapeDescriptor::cpu::BoundingBox boundingBox,
        std::vector<uint32_t>& cumulativeSamplesPerBin,
        std::vector<uint32_t>& pointsInBinMapping,
        ShapeDescriptor::cpu::int3 binCounts,
        double binSize,
        float countRadius) {
    for(uint32_t pointIndex = 0; pointIndex < pointCloud.pointCount; pointIndex++) {
        ShapeDescriptor::cpu::float3 referencePoint = pointCloud.vertices[pointIndex];

        ShapeDescriptor::cpu::float3 referencePointBoundsMin =
                referencePoint - ShapeDescriptor::cpu::float3{countRadius, countRadius, countRadius} - boundingBox.min;
        ShapeDescriptor::cpu::float3 referencePointBoundsMax =
                referencePoint + ShapeDescriptor::cpu::float3{countRadius, countRadius, countRadius} - boundingBox.min;

        // Ensure coordinates range between 0 and length-1
        ShapeDescriptor::cpu::int3 minBinIndices = {
                std::min(std::max(int(referencePointBoundsMin.x / binSize) - 1, 0), binCounts.x - 1),
                std::min(std::max(int(referencePointBoundsMin.y / binSize) - 1, 0), binCounts.y - 1),
                std::min(std::max(int(referencePointBoundsMin.z / binSize) - 1, 0), binCounts.z - 1)
        };

        ShapeDescriptor::cpu::int3 maxBinIndices = {
                std::min(std::max(int(referencePointBoundsMax.x / binSize) + 1, 0), binCounts.x - 1),
                std::min(std::max(int(referencePointBoundsMax.y / binSize) + 1, 0), binCounts.y - 1),
                std::min(std::max(int(referencePointBoundsMax.z / binSize) + 1, 0), binCounts.z - 1)
        };

        assert(minBinIndices.x < binCounts.x);
        assert(minBinIndices.y < binCounts.y);
        assert(minBinIndices.z < binCounts.z);
        assert(maxBinIndices.x < binCounts.x);
        assert(maxBinIndices.y < binCounts.y);
        assert(maxBinIndices.z < binCounts.z);

        for (int binZ = minBinIndices.z; binZ <= maxBinIndices.z; binZ++) {
            for (int binY = minBinIndices.y; binY <= maxBinIndices.y; binY++) {
                unsigned int startTableIndex = computeBinIndex({minBinIndices.x, binY, binZ}, binCounts);
                unsigned int endTableIndex = computeBinIndex({maxBinIndices.x, binY, binZ}, binCounts) + 1;

                unsigned int startVertexIndex = cumulativeSamplesPerBin.at(startTableIndex);
                unsigned int endVertexIndex = 0;

                if (endTableIndex < binCounts.x * binCounts.y * binCounts.z - 1) {
                    endVertexIndex = cumulativeSamplesPerBin.at(endTableIndex);
                } else {
                    endVertexIndex = pointCloud.pointCount;
                }

                assert(startVertexIndex <= endVertexIndex);
                assert(startVertexIndex <= pointCloud.pointCount);
                assert(endVertexIndex <= pointCloud.pointCount);

                for (unsigned int samplePointIndex = startVertexIndex; samplePointIndex < endVertexIndex; samplePointIndex++) {
                    uint32_t mappedIndex = pointsInBinMapping.at(samplePointIndex);
                    if (mappedIndex == pointIndex) {
                        pointDensityArray.at(pointIndex)++;
                        continue;
                    }

                    ShapeDescriptor::cpu::float3 samplePoint = pointCloud.vertices[mappedIndex];
                    ShapeDescriptor::cpu::float3 delta = samplePoint - referencePoint;
                    float distanceToPoint = length(delta);
                    if (distanceToPoint <= countRadius) {
                        pointDensityArray.at(pointIndex)++;
                    }
                }
            }
        }
    }
}

void computePointMapping(ShapeDescriptor::cpu::PointCloud cloud,
                         const std::vector<uint32_t>& cumulativeSamplesPerBin,
                         ShapeDescriptor::cpu::int3 binCounts,
                         double binSize,
                         ShapeDescriptor::cpu::BoundingBox boundingBox,
                        std::vector<uint32_t>& pointsInBinMapping) {
    std::vector<uint32_t> nextIndices = cumulativeSamplesPerBin;
    // The last bin should start counting down from the length of the list. This facilitates that we can request index i + 1
    nextIndices.push_back(cloud.pointCount);

    for(uint32_t vertexIndex = 0; vertexIndex < cloud.pointCount; vertexIndex++) {
        ShapeDescriptor::cpu::float3 vertex = cloud.vertices[vertexIndex];

        uint32_t indexTableIndex = computeBinIndex(vertex, boundingBox, binSize, binCounts);

        assert(indexTableIndex < binCounts.x * binCounts.y * binCounts.z);

        // Reserve next spot
        uint32_t& targetIndex = nextIndices.at(indexTableIndex + 1);
        targetIndex--;
        pointsInBinMapping.at(targetIndex) = vertexIndex;
    }

}

std::vector<unsigned int> ShapeDescriptor::computePointDensities(
        float pointDensityRadius, ShapeDescriptor::cpu::PointCloud pointCloud) {

    // 1. Compute bounding box
    ShapeDescriptor::cpu::BoundingBox boundingBox = ShapeDescriptor::computeBoundingBox(pointCloud);

    // 2. Allocate index array for boxes of radius x radius x radius
    float3 boundingBoxSize = boundingBox.max - boundingBox.min;
    double boundingBoxMax  = std::max(std::max(boundingBoxSize.x, boundingBoxSize.y), boundingBoxSize.z);
    double binSize = boundingBoxMax;
    const double binSizeScaleFactor = 0.66;
    const uint32_t minBinCount = 10000;

    ShapeDescriptor::cpu::int3 binCounts;
    int totalBinCount = binCounts.x * binCounts.y * binCounts.z;
    // Handle malformed meshes
    if(boundingBoxSize.x == 0 || boundingBoxSize.y == 0 || boundingBoxSize.z == 0) {
        binCounts = {1, 1, 1};
        totalBinCount = binCounts.x * binCounts.y * binCounts.z;
    } else {

        while (totalBinCount < minBinCount) {
            binSize *= binSizeScaleFactor;
            binCounts = {int(boundingBoxSize.x / binSize) + 1,
                         int(boundingBoxSize.y / binSize) + 1,
                         int(boundingBoxSize.z / binSize) + 1};
            binCounts.x = std::max(binCounts.x, 1);
            binCounts.y = std::max(binCounts.y, 1);
            binCounts.z = std::max(binCounts.z, 1);
            totalBinCount = binCounts.x * binCounts.y * binCounts.z;
        }
    }

    std::vector<uint32_t> cumulativeSamplesPerBin(totalBinCount);

    // 3. Counting occurrences for each box
    countBinContents(pointCloud, cumulativeSamplesPerBin, boundingBox, binCounts, binSize);

    // 4. Compute cumulative indices
    // Single threaded, because there aren't all that many bins, and you don't win much by parallelising it anyway
    countCumulativeBinIndices(cumulativeSamplesPerBin, binCounts, pointCloud.pointCount);

    // 6. Move points into respective bins
    std::vector<uint32_t> pointsInBinMapping(pointCloud.pointCount);
    computePointMapping(pointCloud, cumulativeSamplesPerBin, binCounts, binSize, boundingBox, pointsInBinMapping);

    // 8. Count nearby points using new array and its index structure
    std::vector<uint32_t> pointCountArray (pointCloud.pointCount);
    computePointCounts(pointCountArray, pointCloud, boundingBox, cumulativeSamplesPerBin, pointsInBinMapping, binCounts, binSize, pointDensityRadius);

    return pointCountArray;
}