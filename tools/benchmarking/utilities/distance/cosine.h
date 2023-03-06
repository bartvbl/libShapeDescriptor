#pragma once
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <shapeDescriptor/cpu/quickIntersectionCountImageGenerator.h>
#include <shapeDescriptor/cpu/spinImageGenerator.h>
#include <shapeDescriptor/gpu/fastPointFeatureHistogramGenerator.cuh>
#include <shapeDescriptor/gpu/3dShapeContextSearcher.cuh>
#include <benchmarking/utilities/distance/generateFakeMetadata.h>
#include <vector>
#include <variant>
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
            double cosineSimilarity(ShapeDescriptor::RICIDescriptor dOne, ShapeDescriptor::RICIDescriptor dTwo);
            double cosineSimilarity(ShapeDescriptor::QUICCIDescriptor dOne, ShapeDescriptor::QUICCIDescriptor dTwo);
            double cosineSimilarity(ShapeDescriptor::SpinImageDescriptor dOne, ShapeDescriptor::SpinImageDescriptor dTwo);
            double cosineSimilarity(ShapeDescriptor::ShapeContextDescriptor dOne, ShapeDescriptor::ShapeContextDescriptor dTwo);
            double cosineSimilarity(ShapeDescriptor::FPFHDescriptor dOne, ShapeDescriptor::FPFHDescriptor dTwo);
        }
    }
}