#include "SpinImageSizeCalculator.h"


float computeSpinImagePixelSize(Mesh mesh) {

    float3 meshDimensions = mesh.boundingBoxMax - mesh.boundingBoxMin;

    float cubeSize = std::cbrt(meshDimensions.x * meshDimensions.y * meshDimensions.z);

    float pixelSize = (cubeSize * 0.5f) / (float(spinImageWidthPixels));

    return pixelSize;
}