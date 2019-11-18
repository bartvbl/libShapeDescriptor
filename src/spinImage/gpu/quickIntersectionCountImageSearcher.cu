#include "quickIntersectionCountImageSearcher.cuh"
#include <spinImage/gpu/quickIntersectionCountImageGenerator.cuh>

SpinImage::array<SpinImage::gpu::QUICCISearchResults> SpinImage::gpu::findQUICCImagesInHaystack(
        SpinImage::gpu::QUICCIImages device_needleDescriptors,
        size_t needleImageCount,
        SpinImage::gpu::QUICCIImages device_haystackDescriptors,
        size_t haystackImageCount) {

}

SpinImage::array<unsigned int> SpinImage::gpu::computeQUICCImageSearchResultRanks(
        SpinImage::gpu::QUICCIImages device_needleDescriptors,
        size_t needleImageCount,
        SpinImage::gpu::QUICCIImages device_haystackDescriptors,
        size_t haystackImageCount,
        SpinImage::debug::QUICCISearchRunInfo* runInfo) {

}