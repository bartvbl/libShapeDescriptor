#pragma once
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <shapeDescriptor/cpu/quickIntersectionCountImageGenerator.h>
#include <shapeDescriptor/cpu/spinImageGenerator.h>
#include <shapeDescriptor/gpu/fastPointFeatureHistogramGenerator.cuh>
#include <shapeDescriptor/gpu/3dShapeContextSearcher.cuh>
#include <shapeDescriptor/cpu/types/array.h>

extern int radialIntersectionLength;
extern int quickIntersectionLength;
extern int spinImageLength;
extern int shapeContextLength;
extern int fpfhLength;

namespace Benchmarking
{
    namespace utilities
    {
        namespace distance
        {
            double euclidianSimilarity(ShapeDescriptor::RICIDescriptor dOne, ShapeDescriptor::RICIDescriptor dTwo);
            double euclidianSimilarity(ShapeDescriptor::QUICCIDescriptor dOne, ShapeDescriptor::QUICCIDescriptor dTwo);
            double euclidianSimilarity(ShapeDescriptor::SpinImageDescriptor dOne, ShapeDescriptor::SpinImageDescriptor dTwo);
            double euclidianSimilarity(ShapeDescriptor::ShapeContextDescriptor dOne, ShapeDescriptor::ShapeContextDescriptor dTwo);
            double euclidianSimilarityOffset(ShapeDescriptor::ShapeContextDescriptor dOne, ShapeDescriptor::ShapeContextDescriptor dTwo, int sliceOffset);
            double euclidianSimilarity(ShapeDescriptor::FPFHDescriptor dOne, ShapeDescriptor::FPFHDescriptor dTwo);
        }
    }
}