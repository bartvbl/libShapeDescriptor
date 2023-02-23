#include "euclidian.h"
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <shapeDescriptor/cpu/quickIntersectionCountImageGenerator.h>
#include <shapeDescriptor/cpu/spinImageGenerator.h>
#include <shapeDescriptor/gpu/fastPointFeatureHistogramGenerator.cuh>
#include <shapeDescriptor/gpu/3dShapeContextSearcher.cuh>
#include <benchmarking/utilities/distance/generateFakeMetadata.h>
#include <shapeDescriptor/cpu/types/array.h>
#include <math.h>

// RICI
double Benchmarking::utilities::distance::euclidianSimilarity(ShapeDescriptor::RICIDescriptor dOne, ShapeDescriptor::RICIDescriptor dTwo)
{
    double distance = 0;

    for (int i = 0; i < (int)(spinImageWidthPixels * spinImageWidthPixels); i++)
    {
        double diff = (double)dOne.contents[i] - (double)dTwo.contents[i];
        distance += pow(diff, 2);
    }

    double similarity = 1.0 / (1.0 + sqrt(distance));

    return similarity;
}

// QUICCI
double Benchmarking::utilities::distance::euclidianSimilarity(ShapeDescriptor::QUICCIDescriptor dOne, ShapeDescriptor::QUICCIDescriptor dTwo)
{
    double distance = 0;

    for (int i = 0; i < (int)(ShapeDescriptor::QUICCIDescriptorLength); i++)
    {
        double diff = (double)dOne.contents[i] - (double)dTwo.contents[i];
        distance += pow(diff, 2);
    }

    double similarity = 1.0 / (1.0 + sqrt(distance));

    return similarity;
}

// Spin Image
double Benchmarking::utilities::distance::euclidianSimilarity(ShapeDescriptor::SpinImageDescriptor dOne, ShapeDescriptor::SpinImageDescriptor dTwo)
{
    double distance = 0;

    for (int i = 0; i < (int)(spinImageWidthPixels * spinImageWidthPixels); i++)
    {
        double diff = (double)dOne.contents[i] - (double)dTwo.contents[i];
        distance += pow(diff, 2);
    }

    double similarity = 1.0 / (1.0 + sqrt(distance));

    return similarity;
}

// 3D Shape Context
double Benchmarking::utilities::distance::euclidianSimilarity(ShapeDescriptor::ShapeContextDescriptor dOne, ShapeDescriptor::ShapeContextDescriptor dTwo)
{
    double distance = 0;

    for (int i = 0; i < (int)(SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT * SHAPE_CONTEXT_VERTICAL_SLICE_COUNT * SHAPE_CONTEXT_LAYER_COUNT); i++)
    {
        double diff = (double)dOne.contents[i] - (double)dTwo.contents[i];
        distance += pow(diff, 2);
    }

    double similarity = 1.0 / (1.0 + sqrt(distance));

    return similarity;
}

// FPFH
double Benchmarking::utilities::distance::euclidianSimilarity(ShapeDescriptor::FPFHDescriptor dOne, ShapeDescriptor::FPFHDescriptor dTwo)
{
    double distance = 0;

    for (int i = 0; i < (int)(3 * FPFH_BINS_PER_FEATURE); i++)
    {
        double diff = (double)dOne.contents[i] - (double)dTwo.contents[i];
        distance += pow(diff, 2);
    }

    double similarity = 1.0 / (1.0 + sqrt(distance));

    return similarity;
}
