#pragma once
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <shapeDescriptor/cpu/quickIntersectionCountImageGenerator.h>
#include <shapeDescriptor/cpu/spinImageGenerator.h>
#include <shapeDescriptor/gpu/fastPointFeatureHistogramGenerator.cuh>
#include <shapeDescriptor/gpu/3dShapeContextSearcher.cuh>
#include <benchmarking/utilities/distance/generateFakeMetadata.h>
#include <shapeDescriptor/cpu/types/array.h>

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
            double euclidianSimilarity(ShapeDescriptor::FPFHDescriptor dOne, ShapeDescriptor::FPFHDescriptor dTwo);
        }
    }
}