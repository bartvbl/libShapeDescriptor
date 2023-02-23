#include "cosine.h"
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <shapeDescriptor/cpu/quickIntersectionCountImageGenerator.h>
#include <shapeDescriptor/cpu/spinImageGenerator.h>
#include <shapeDescriptor/gpu/3dShapeContextGenerator.cuh>
#include <shapeDescriptor/gpu/fastPointFeatureHistogramGenerator.cuh>
#include <benchmarking/utilities/distance/generateFakeMetadata.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <variant>

// RICI
double Benchmarking::utilities::distance::cosineSimilarity(ShapeDescriptor::RICIDescriptor dOne, ShapeDescriptor::RICIDescriptor dTwo)
{
    double dot = 0;
    double denominationA = 0;
    double denominationB = 0;

    for (int i = 0; i < (int)(spinImageWidthPixels * spinImageWidthPixels); i++)
    {
        dot += dOne.contents[i] * dTwo.contents[i];
        denominationA += pow(dOne.contents[i], 2);
        denominationB += pow(dTwo.contents[i], 2);
    }

    double similarity = dot / sqrt(denominationA * denominationB);

    return isnan(similarity) ? 0 : similarity;
}

// QUICCI
double Benchmarking::utilities::distance::cosineSimilarity(ShapeDescriptor::QUICCIDescriptor dOne, ShapeDescriptor::QUICCIDescriptor dTwo)
{
    double dot = 0;
    double denominationA = 0;
    double denominationB = 0;

    for (int i = 0; i < (int)(ShapeDescriptor::QUICCIDescriptorLength); i++)
    {
        dot += (double)dOne.contents[i] * (double)dTwo.contents[i];
        denominationA += pow((double)dOne.contents[i], 2);
        denominationB += pow((double)dTwo.contents[i], 2);
    }

    double similarity = dot / sqrt(denominationA * denominationB);

    return isnan(similarity) ? 0 : similarity;
}

// Spin Image
double Benchmarking::utilities::distance::cosineSimilarity(ShapeDescriptor::SpinImageDescriptor dOne, ShapeDescriptor::SpinImageDescriptor dTwo)
{
    double dot = 0;
    double denominationA = 0;
    double denominationB = 0;

    for (int i = 0; i < (int)(spinImageWidthPixels * spinImageWidthPixels); i++)
    {
        dot += (double)dOne.contents[i] * (double)dTwo.contents[i];
        denominationA += pow((double)dOne.contents[i], 2);
        denominationB += pow((double)dTwo.contents[i], 2);
    }

    double similarity = dot / sqrt(denominationA * denominationB);

    return isnan(similarity) ? 0 : similarity;
}

// 3D Shape Context
double Benchmarking::utilities::distance::cosineSimilarity(ShapeDescriptor::ShapeContextDescriptor dOne, ShapeDescriptor::ShapeContextDescriptor dTwo)
{
    double dot = 0;
    double denominationA = 0;
    double denominationB = 0;

    for (int i = 0; i < (int)(SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT * SHAPE_CONTEXT_VERTICAL_SLICE_COUNT * SHAPE_CONTEXT_LAYER_COUNT); i++)
    {
        dot += (double)dOne.contents[i] * (double)dTwo.contents[i];
        denominationA += pow((double)dOne.contents[i], 2);
        denominationB += pow((double)dTwo.contents[i], 2);
    }

    double similarity = dot / sqrt(denominationA * denominationB);

    return isnan(similarity) ? 0 : similarity;
}

// FPFH
double Benchmarking::utilities::distance::cosineSimilarity(ShapeDescriptor::FPFHDescriptor dOne, ShapeDescriptor::FPFHDescriptor dTwo)
{
    double dot = 0;
    double denominationA = 0;
    double denominationB = 0;

    for (int i = 0; i < (int)(3 * FPFH_BINS_PER_FEATURE); i++)
    {
        dot += (double)dOne.contents[i] * (double)dTwo.contents[i];
        denominationA += pow((double)dOne.contents[i], 2);
        denominationB += pow((double)dTwo.contents[i], 2);
    }

    double similarity = dot / sqrt(denominationA * denominationB);

    return isnan(similarity) ? 0 : similarity;
}
