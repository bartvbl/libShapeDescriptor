#include <fast-lzma2.h>
#include <shapeDescriptor/shapeDescriptor.h>

const int LZMA2_COMPRESSION_LEVEL = 9;

size_t ShapeDescriptor::compressBytes(
        void *outputBuffer, size_t outputBufferCapacity,
        const void *inputBuffer, size_t inputBufferSize) {
    return FL2_compress(
            (void*) outputBuffer, outputBufferCapacity,
            (void*) inputBuffer, inputBufferSize,
            LZMA2_COMPRESSION_LEVEL);
}

size_t ShapeDescriptor::compressBytesMultithreaded(
        void *outputBuffer, size_t outputBufferCapacity,
        const void *inputBuffer, size_t inputBufferSize,
        unsigned int numThreads) {
    return FL2_compressMt(
            (void*) outputBuffer, outputBufferCapacity,
            (void*) inputBuffer, inputBufferSize,
            LZMA2_COMPRESSION_LEVEL, numThreads);
}

size_t ShapeDescriptor::decompressBytes(
        void *outputBuffer, size_t outputBufferCapacity,
        const void *inputBuffer, size_t inputBufferCapacity) {
    return FL2_decompress(
            (void*) outputBuffer, outputBufferCapacity,
            (void*) inputBuffer, inputBufferCapacity);
}

size_t ShapeDescriptor::decompressBytesMultithreaded(
        void *outputBuffer, size_t outputBufferCapacity,
        const void *inputBuffer, size_t inputBufferCapacity,
        unsigned int numThreads) {
    return FL2_decompressMt(
            (void*) outputBuffer, outputBufferCapacity,
            (void*) inputBuffer, inputBufferCapacity,
            numThreads);
}

size_t ShapeDescriptor::computeMaxCompressedBufferSize(size_t inputBufferSize) {
    return FL2_compressBound(inputBufferSize);
}
