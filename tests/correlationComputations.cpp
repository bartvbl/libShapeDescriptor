#include "correlationComputations.h"
#include <catch2/catch.hpp>
#include <shapeSearch/common/buildSettings/derivedBuildSettings.h>
#include <shapeSearch/common/types/array.h>
#include <shapeSearch/libraryBuildSettings.h>
#include <cuda_runtime.h>
#include <shapeSearch/utilities/CUDAContextCreator.h>
#include <nvidia/helper_cuda.h>

template<typename pixelType>
array<pixelType> generateRepeatingTemplateImage(
        pixelType patternPart0,
        pixelType patternPart1,
        pixelType patternPart2,
        pixelType patternPart3,
        pixelType patternPart4,
        pixelType patternPart5,
        pixelType patternPart6,
        pixelType patternPart7) {

    pixelType* image = new pixelType[spinImageWidthPixels * spinImageWidthPixels];

    for(size_t index = 0; index < spinImageWidthPixels * spinImageWidthPixels; index += 8) {
        image[index + 0] = patternPart0;
        image[index + 1] = patternPart1;
        image[index + 2] = patternPart2;
        image[index + 3] = patternPart3;
        image[index + 4] = patternPart4;
        image[index + 5] = patternPart5;
        image[index + 6] = patternPart6;
        image[index + 7] = patternPart7;
    }

    size_t bufferSize = sizeof(pixelType) * spinImageWidthPixels * spinImageWidthPixels;

    array<pixelType> device_images;
    device_images.length = spinImageWidthPixels * spinImageWidthPixels;
    checkCudaErrors(cudaMalloc(&device_images.content, bufferSize));
    checkCudaErrors(cudaMemcpy(device_images.content, image, bufferSize, cudaMemcpyHostToDevice));

    delete[] image;

    return device_images;
}

TEST_CASE("Correlation computation", "[correlation]") {
    SpinImage::utilities::createCUDAContext();

    SECTION("Equivalent images") {
        array<spinImagePixelType> constantImage =
                generateRepeatingTemplateImage<spinImagePixelType>(0, 1, 0, 1, 0, 1, 0, 1);



        cudaFree(constantImage.content);
    }

}