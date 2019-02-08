#include <shapeSearch/libraryBuildSettings.h>
#include <cmath>
#include "SpinImageSizeCalculator.h"


float computeSpinImagePixelSize(HostMesh mesh) {

    float3_cpu meshDimensions = mesh.boundingBoxMax - mesh.boundingBoxMin;

    float cubeSize = std::cbrt(meshDimensions.x * meshDimensions.y * meshDimensions.z);

    float pixelSize = (cubeSize * 0.5f) / (float(spinImageWidthPixels));

    return pixelSize;
}