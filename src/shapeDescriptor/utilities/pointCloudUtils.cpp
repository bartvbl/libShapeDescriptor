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











inline unsigned int computeJumpTableIndex(ShapeDescriptor::cpu::int3 binIndex, ShapeDescriptor::cpu::int3 binCounts) {
    return binIndex.z * binCounts.x * binCounts.y + binIndex.y * binCounts.x + binIndex.x;
}

void countBinContents(
        ShapeDescriptor::cpu::PointCloud pointCloud,
        std::vector<uint32_t>& indexTable,
        ShapeDescriptor::cpu::BoundingBox boundingBox,
        ShapeDescriptor::cpu::int3 binCounts,
        float binSize) {

    for(uint32_t vertexIndex = 0; vertexIndex < pointCloud.pointCount; vertexIndex++) {
        ShapeDescriptor::cpu::float3 vertex = pointCloud.vertices[vertexIndex];

        ShapeDescriptor::cpu::float3 relativeToBoundingBox = vertex - boundingBox.min;

        ShapeDescriptor::cpu::int3 binIndex = {
                std::min(std::max(int(relativeToBoundingBox.x / binSize), 0), binCounts.x - 1),
                std::min(std::max(int(relativeToBoundingBox.y / binSize), 0), binCounts.y - 1),
                std::min(std::max(int(relativeToBoundingBox.z / binSize), 0), binCounts.z - 1)
        };

        uint32_t indexTableIndex = computeJumpTableIndex(binIndex, binCounts);

        assert(indexTableIndex < binCounts.x * binCounts.y * binCounts.z);

        indexTable[indexTableIndex]++;
    }
}

void countCumulativeBinIndices(std::vector<uint32_t>& indexTable, ShapeDescriptor::cpu::int3 binCounts, unsigned int pointCloudSize) {
    unsigned int cumulativeIndex = 0;
    for(int z = 0; z < binCounts.z; z++) {
        for(int y = 0; y < binCounts.y; y++) {
            for(int x = 0; x < binCounts.x; x++) {
                unsigned int binIndex = computeJumpTableIndex({x, y, z}, binCounts);
                unsigned int binLength = indexTable.at(binIndex);
                indexTable.at(binIndex) = cumulativeIndex;
                cumulativeIndex += binLength;
            }
        }
    }
    assert(cumulativeIndex == pointCloudSize);
}

void rearrangePointCloud(
        ShapeDescriptor::cpu::PointCloud sourcePointCloud,
        ShapeDescriptor::cpu::PointCloud destinationPointCloud,
        ShapeDescriptor::cpu::BoundingBox boundingBox,
        std::vector<uint32_t>& nextIndexEntryTable,
        ShapeDescriptor::cpu::int3 binCounts,
        float binSize) {
    for(uint32_t vertexIndex = 0; vertexIndex < sourcePointCloud.pointCount; vertexIndex++) {
        ShapeDescriptor::cpu::float3 vertex = sourcePointCloud.vertices[vertexIndex];

        ShapeDescriptor::cpu::float3 relativeToBoundingBox = vertex - boundingBox.min;

        ShapeDescriptor::cpu::int3 binIndex = {
                std::min(std::max(int(relativeToBoundingBox.x / binSize), 0), binCounts.x - 1),
                std::min(std::max(int(relativeToBoundingBox.y / binSize), 0), binCounts.y - 1),
                std::min(std::max(int(relativeToBoundingBox.z / binSize), 0), binCounts.z - 1)
        };

        unsigned int indexTableIndex = computeJumpTableIndex(binIndex, binCounts);

        assert(indexTableIndex < binCounts.x * binCounts.y * binCounts.z);

        nextIndexEntryTable.at(indexTableIndex)++;
        unsigned int targetIndex = nextIndexEntryTable.at(indexTableIndex);

        destinationPointCloud.vertices[targetIndex] = vertex;
    }
}

void computePointCounts(
        ShapeDescriptor::cpu::array<uint32_t> pointDensityArray,
        ShapeDescriptor::cpu::PointCloud pointCloud,
        ShapeDescriptor::cpu::BoundingBox boundingBox,
        std::vector<uint32_t>& indexTable,
        ShapeDescriptor::cpu::int3 binCounts,
        float binSize,
        float countRadius) {
    for(uint32_t pointIndex = 0; pointIndex < pointCloud.pointCount; pointIndex++) {
        unsigned int totalPointCount = 0;
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
                unsigned int startTableIndex = computeJumpTableIndex({minBinIndices.x, binY, binZ}, binCounts);
                unsigned int endTableIndex = computeJumpTableIndex({maxBinIndices.x, binY, binZ}, binCounts) + 1;

                unsigned int startVertexIndex = indexTable[startTableIndex];
                unsigned int endVertexIndex = 0;

                if (endTableIndex < binCounts.x * binCounts.y * binCounts.z - 1) {
                    endVertexIndex = indexTable[endTableIndex];
                } else {
                    endVertexIndex = pointCloud.pointCount;
                }

                assert(startVertexIndex <= endVertexIndex);
                assert(startVertexIndex <= pointCloud.pointCount);
                assert(endVertexIndex <= pointCloud.pointCount);

                for (unsigned int samplePointIndex = startVertexIndex; samplePointIndex < endVertexIndex; samplePointIndex++) {
                    if (samplePointIndex == pointIndex) {
                        continue;
                    }

                    ShapeDescriptor::cpu::float3 samplePoint = pointCloud.vertices[samplePointIndex];
                    ShapeDescriptor::cpu::float3 delta = samplePoint - referencePoint;
                    float distanceToPoint = length(delta);
                    if (distanceToPoint <= countRadius) {
                        totalPointCount++;
                    }
                }
            }
        }
        pointDensityArray.content[pointIndex] = totalPointCount;
    }
}

ShapeDescriptor::cpu::array<unsigned int> ShapeDescriptor::computePointDensities(
        float pointDensityRadius, ShapeDescriptor::cpu::PointCloud pointCloud) {

    size_t sampleCount = pointCloud.pointCount;

    // 1. Compute bounding box
    ShapeDescriptor::cpu::BoundingBox boundingBox = ShapeDescriptor::computeBoundingBox(pointCloud);

    // 2. Allocate index array for boxes of radius x radius x radius
    float3 boundingBoxSize = boundingBox.max - boundingBox.min;
    float binSize = std::cbrt(
            (boundingBoxSize.x != 0 ? boundingBoxSize.x : 1) *
            (boundingBoxSize.y != 0 ? boundingBoxSize.y : 1) *
            (boundingBoxSize.z != 0 ? boundingBoxSize.z : 1)) / 50.0f;

    ShapeDescriptor::cpu::int3 binCounts = {int(boundingBoxSize.x / binSize) + 1,
                                            int(boundingBoxSize.y / binSize) + 1,
                                            int(boundingBoxSize.z / binSize) + 1};
    int totalBinCount = binCounts.x * binCounts.y * binCounts.z;
    std::vector<uint32_t> indexTable(totalBinCount);

    // 3. Counting occurrences for each box
    countBinContents(pointCloud, indexTable, boundingBox, binCounts, binSize);

    // 4. Compute cumulative indices
    // Single threaded, because there aren't all that many bins, and you don't win much by parallelising it anyway
    countCumulativeBinIndices(indexTable, binCounts, pointCloud.pointCount);

    // 5. Allocate temporary point cloud (vertices only)
    ShapeDescriptor::cpu::PointCloud tempPointCloud = pointCloud.clone();

    // 6. Move points into respective bins
    std::vector<uint32_t> nextIndexTableEntries(totalBinCount);
    std::copy(indexTable.begin(), indexTable.end(), nextIndexTableEntries.begin());

    rearrangePointCloud(
            tempPointCloud, pointCloud,
            boundingBox,
            nextIndexTableEntries,
            binCounts, binSize);

    // 7. Delete temporary vertex buffer
    ShapeDescriptor::free(tempPointCloud);

    // 8. Count nearby points using new array and its index structure
    ShapeDescriptor::cpu::array<uint32_t> pointCountArray (sampleCount);
    computePointCounts(pointCountArray, pointCloud, boundingBox, indexTable, binCounts, binSize, pointDensityRadius);

    return pointCountArray;
}