#pragma once

#include <cstddef>

// Thin wrapper around the compression library used in the project to allow it to be swapped out easily if necessary
// Each function takes a buffer pair, and returns the size of the produced (de)compressed buffer

namespace ShapeDescriptor {
    namespace utilities {
        size_t compressBytes(void* outputBuffer, size_t outputBufferCapacity,
                           const void* inputBuffer, size_t inputBufferSize);
        size_t compressBytesMultithreaded(void* outputBuffer, size_t outputBufferCapacity,
                                       const void* inputBuffer, size_t inputBufferSize,
                                       unsigned numThreads);
        size_t decompressBytes(void* outputBuffer, size_t outputBufferCapacity,
                             const void* inputBuffer, size_t inputBufferCapacity);
        size_t decompressBytesMultithreaded(void* outputBuffer, size_t outputBufferCapacity,
                                          const void* inputBuffer, size_t inputBufferCapacity,
                                          unsigned int numThreads);
        size_t computeMaxCompressedBufferSize(size_t inputBufferSize);
    }
}