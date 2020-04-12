#pragma once

#include <cstddef>
#include <fast-lzma2.h>
#include <fstream>
#include <cassert>
#include <array>

namespace SpinImage {
    namespace utilities {
        template<typename DataType, size_t internalBufferCount> class FileDecompressionStream {
        private:
            FL2_DStream* decompressionStream;
            std::fstream* fileStream;

            size_t totalCompressedSize = 0;
            size_t totalDecompressedSize = 0;

            const size_t internalBufferSize = internalBufferCount * sizeof(DataType);

            char cBuffer[internalBufferCount * sizeof(DataType)];
            FL2_inBuffer compressedBuffer = {cBuffer, internalBufferSize, internalBufferSize};

            size_t totalReadCompressedBytes = 0;
            size_t totalReturnedDecompressedBytes = 0;

        public:
            FileDecompressionStream(std::fstream* stream,
                    size_t totalCompressedBufferSize, size_t totalDecompressedElementCount) {
                decompressionStream = FL2_createDStream();
                FL2_initDStream(decompressionStream);
                fileStream = stream;
                totalCompressedSize = totalCompressedBufferSize;
                totalDecompressedSize = totalDecompressedElementCount * sizeof(DataType);
            }

            ~FileDecompressionStream() {
                FL2_freeDStream(decompressionStream);
            }

            bool isDepleted() {
                return totalReadCompressedBytes == totalCompressedSize &&
                        totalReturnedDecompressedBytes == totalDecompressedSize;
            }

            size_t read(std::array<DataType, internalBufferCount> &decompressedDataBuffer) {
                FL2_outBuffer uncompressedBuffer = {decompressedDataBuffer.data(), internalBufferSize, 0};

                // Keep going until input buffer is full, or no more data is available
                while(uncompressedBuffer.pos < internalBufferSize
                      && (totalReturnedDecompressedBytes + uncompressedBuffer.pos) < totalDecompressedSize) {

                    // If the compressed data buffer has been used up, refill it
                    if (compressedBuffer.pos == compressedBuffer.size) {
                        // Read a full buffer, or whatever remains of the compressed input data
                        size_t numberOfBytesToRead =
                                std::min<size_t>(totalCompressedSize - totalReadCompressedBytes,
                                                 internalBufferSize);
                        compressedBuffer.size = numberOfBytesToRead;
                        fileStream->read(cBuffer, numberOfBytesToRead);
                        compressedBuffer.pos = 0;

                        totalReadCompressedBytes += numberOfBytesToRead;
                    }

                    FL2_decompressStream(decompressionStream, &uncompressedBuffer, &compressedBuffer);
                }

                totalReturnedDecompressedBytes += uncompressedBuffer.pos;

                // Ensure uncompressed buffer is either full, or contains the last remainder of the decompressed data
                assert(uncompressedBuffer.pos == internalBufferSize
                       || totalReturnedDecompressedBytes == totalDecompressedSize);

                // Ensure contents of uncompressed buffer contain complete data structures (sanity check)
                assert(uncompressedBuffer.pos % sizeof(DataType) == 0);

                // Return the number of decompressed elements in the output buffer
                return uncompressedBuffer.pos / sizeof(DataType);
            }
        };
    }
}