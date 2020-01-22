#include <cmath>
#include <iostream>

#define SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT 32
#define SHAPE_CONTEXT_VERTICAL_SLICE_COUNT 15
#define SHAPE_CONTEXT_LAYER_COUNT 12



struct float2 {
    float x;
    float y;
};

struct float3 {
    float x;
    float y;
    float z;
};

const float3 vertex = {0, 0, 0};
const float3 normal = {0, 1, 0};

const float3 samplePoint = {3, 0, 5};
const float minSupportRadius = 0.1;
const float maxSupportRadius = 12;

const unsigned int pointDensity = 12;

struct short3 {
    short x;
    short y;
    short z;
};

bool operator==(float3 &a, float3 &b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

bool operator==(const float3 &a, const float3 &b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

bool operator==(float2 &a, float2 &b) {
    return a.x == b.x && a.y == b.y;
}

bool operator==(const float2 &a, const float2 &b) {
    return a.x == b.x && a.y == b.y;
}

float3 cross(float3 a, float3 b)
{
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}

float length(float3 a) {
    return std::sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}

float length(float2 a) {
    return std::sqrt(a.x*a.x + a.y*a.y);
}

float3 operator-(float3 a, float3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

void operator/=(float2 &a, float b)
{
    a.x /= b;
    a.y /= b;
}

float absoluteAngle(float y, float x) {
    float absoluteAngle = std::atan2(y, x);
    return absoluteAngle < 0 ? absoluteAngle + (2.0f * float(M_PI)) : absoluteAngle;
}

float computeLayerDistance(float minSupportRadius, float maxSupportRadius, short layerIndex) {
    return std::exp(
            std::log(minSupportRadius)
            + (float(layerIndex) / float(SHAPE_CONTEXT_LAYER_COUNT))
            * std::log(float(maxSupportRadius) / float(minSupportRadius))
        );
}

float computeWedgeSegmentVolume(short verticalBinIndex, float radius) {
    const float verticalAngleStep = 1.0f / float(SHAPE_CONTEXT_VERTICAL_SLICE_COUNT);
    float binStartAngle = float(verticalBinIndex) * verticalAngleStep;
    float binEndAngle = float(verticalBinIndex + 1) * verticalAngleStep;

    float scaleFraction = (2.0f * float(M_PI) * radius * radius * radius)
                        / (3.0f * float(SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT));
    return scaleFraction * (std::cos(binStartAngle) - std::cos(binEndAngle));
}

float computeBinVolume(short verticalBinIndex, short layerIndex, float minSupportRadius, float maxSupportRadius) {
    // The wedge segment computation goes all the way from the center to the edge of the sphere
    // Since we also have a minimum support radius, we need to cut out the volume of the centre part
    float binEndRadius = computeLayerDistance(minSupportRadius, maxSupportRadius, layerIndex + 1);
    float binStartRadius = computeLayerDistance(minSupportRadius, maxSupportRadius, layerIndex);

    float largeSupportRadiusVolume = computeWedgeSegmentVolume(verticalBinIndex, binEndRadius);
    float smallSupportRadiusVolume = computeWedgeSegmentVolume(verticalBinIndex, binStartRadius);

    return largeSupportRadiusVolume - smallSupportRadiusVolume;
}

float toDegrees(float a) {
    return a * (180.0f / M_PI);
}

std::ostream& operator << (std::ostream &o, const float3& p)
{
    return o << "(" << p.x << ", " << p.y << ", " << p.z << ")";
}

std::ostream& operator << (std::ostream &o, const float2& p)
{
    return o << "(" << p.x << ", " << p.y << ")";
}

int main(int argc, char** argv) {
    std::cout << "Input vertex: " << vertex << std::endl;
    std::cout << "Input normal: " << normal << std::endl;
    std::cout << "Input sample point: " << samplePoint << std::endl;
    std::cout << "Min support radius: " << minSupportRadius << std::endl;
    std::cout << "Max support radius: " << maxSupportRadius << std::endl;
    std::cout << "Point density: " << pointDensity << std::endl << std::endl;

    for(int i = 0; i < 360; i++) {
        float cosine = std::cos(float(i) * M_PI / 180.0f);
        float sine = std::sin(float(i) * M_PI / 180.0f);
        std::cout << i << " -> " << toDegrees(absoluteAngle(sine, cosine));
        float horizontalAngle = absoluteAngle(sine, cosine);
        short horizontalIndex =
                unsigned((horizontalAngle / (2.0f * float(M_PI))) *
                         float(SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT))
                % SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT;
        std::cout << "\t" << toDegrees(horizontalAngle) << "/" << horizontalIndex;

        if(cosine >= 0) {
            float verticalAngle = std::fmod(absoluteAngle(sine, cosine) + (float(M_PI) / 2.0f), 2.0f * float(M_PI));
            short verticalIndex =
                    unsigned((verticalAngle / M_PI) * float(SHAPE_CONTEXT_VERTICAL_SLICE_COUNT))
                    % SHAPE_CONTEXT_VERTICAL_SLICE_COUNT;
            std::cout << "\t" << toDegrees(verticalAngle) << "/" << verticalIndex;
        } else {
            std::cout << "\t\t";
        }
        std::cout << "\t" << float2{cosine, sine} << std::endl;

    }

    // First, we align the input vertex with the descriptor's coordinate system
    float3 arbitraryAxis = {0, 0, 1};
    if (normal == arbitraryAxis) {
        arbitraryAxis = {1, 0, 0};
    }

    const float3 referenceXAxis = cross(arbitraryAxis, normal);
    const float3 referenceYAxis = cross(referenceXAxis, normal);

    std::cout << "Reference X axis: " << referenceXAxis << std::endl;
    std::cout << "Reference Y axis: " << referenceYAxis << std::endl;
    std::cout << "Reference Z axis: " << normal << std::endl << std::endl;

    // 1. Compute bin indices

    const float3 translated = samplePoint - vertex;

    std::cout << "Translated point: " << translated << std::endl;

    // Only include vertices which are within the support radius
    float distanceToVertex = length(translated);
    if (distanceToVertex < minSupportRadius || distanceToVertex > maxSupportRadius) {
        std::cout << "Exited: not in support sphere" << std::endl;
        return 0;
    }
    std::cout << "Distance to vertex: " << distanceToVertex << std::endl;

    // Transforming descriptor coordinate system to the origin
    // In the new system, 'z' is 'up'
    const float3 relativeSamplePoint = {
            referenceXAxis.x * translated.x + referenceXAxis.y * translated.y + referenceXAxis.z * translated.z,
            referenceYAxis.x * translated.x + referenceYAxis.y * translated.y + referenceYAxis.z * translated.z,
            normal.x * translated.x + normal.y * translated.y + normal.z * translated.z,
    };

    std::cout << "Relative point: " << relativeSamplePoint << std::endl << std::endl;

    float2 horizontalDirection = {relativeSamplePoint.x, relativeSamplePoint.y};
    float2 verticalDirection = {length(horizontalDirection), relativeSamplePoint.z};

    std::cout << "Horizontal direction: " << horizontalDirection << std::endl;
    std::cout << "Vertical direction: " << verticalDirection << std::endl;

    if (horizontalDirection == float2{0, 0}) {
        // special case, will result in an angle of 0
        horizontalDirection = {1, 0};
    }

    // normalise direction vector
    horizontalDirection /= length(horizontalDirection);
    verticalDirection /= length(verticalDirection);

    std::cout << "Normalised horizontal direction: " << horizontalDirection << std::endl;
    std::cout << "Normalised vertical direction: " << verticalDirection << std::endl << std::endl;

    float horizontalAngle = absoluteAngle(horizontalDirection.y, horizontalDirection.x);
    short horizontalIndex =
            unsigned((horizontalAngle / (2.0f * float(M_PI))) *
                     float(SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT))
            % SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT;
    std::cout << "Horizontal angle: " << toDegrees(horizontalAngle) << std::endl;
    std::cout << "Horizontal index: " << horizontalIndex << std::endl;

    float verticalAngle = std::fmod(absoluteAngle(verticalDirection.y, verticalDirection.x) + (float(M_PI) / 2.0f), 2.0f * float(M_PI));
    short verticalIndex =
            unsigned((verticalAngle / M_PI) * float(SHAPE_CONTEXT_VERTICAL_SLICE_COUNT))
            % SHAPE_CONTEXT_VERTICAL_SLICE_COUNT;
    std::cout << "Vertical angle: " << toDegrees(verticalAngle) << std::endl;
    std::cout << "Vertical index: " << verticalIndex << std::endl << std::endl;

    float sampleDistance = length(relativeSamplePoint);
    short layerIndex = 0;
    std::cout << "Sample distance: " << sampleDistance << std::endl;

    // Recomputing logarithms is still preferable over doing memory transactions for each of them
    for (; layerIndex <= SHAPE_CONTEXT_LAYER_COUNT; layerIndex++) {
        float nextSliceEnd = computeLayerDistance(minSupportRadius, maxSupportRadius, layerIndex+1);
        std::cout << "   Slice " << layerIndex << " ends at " << nextSliceEnd << std::endl;
        if (sampleDistance < nextSliceEnd) {
            break;
        }
    }
    std::cout << "Layer index: " << layerIndex << std::endl << std::endl;

    short3 binIndex = {horizontalIndex, verticalIndex, layerIndex};

    // 2. Compute sample weight
    float binVolume = computeBinVolume(binIndex.y, binIndex.z, minSupportRadius, maxSupportRadius);
    float sampleWeight = 1.0f / pointDensity * std::cbrt(binVolume);
    std::cout << "Bin volume: " << binVolume << std::endl;
    std::cout << "Sample weight: " << sampleWeight << std::endl;

    // 3. Increment appropriate bin
    unsigned int index =
            binIndex.x * SHAPE_CONTEXT_LAYER_COUNT * SHAPE_CONTEXT_VERTICAL_SLICE_COUNT +
            binIndex.y * SHAPE_CONTEXT_LAYER_COUNT +
            binIndex.z;
    std::cout << "------------" << std::endl;
    std::cout << "Bin x: " << binIndex.x << std::endl;
    std::cout << "Bin y: " << binIndex.y << std::endl;
    std::cout << "Bin z: " << binIndex.z << std::endl;
    std::cout << "Contribution: " << sampleWeight << std::endl;

    return 0;
}