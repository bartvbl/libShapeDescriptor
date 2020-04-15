#pragma once

#include <fstream>
#include <array>
#include <fast-lzma2.h>
#include <cassert>

namespace SpinImage {
    namespace utilities {
        template<typename DataType, size_t internalBufferCount> class FileCompressionStream {
        private:
            FL2_CStream* compressionStream;
            std::fstream* fileStream;

            size_t totalCompressedSize = 0;

            static const size_t internalBufferSize = internalBufferCount * sizeof(DataType);

            unsigned char cBuffer[internalBufferSize];
        public:
            FileCompressionStream(std::fstream* stream) {
                fileStream = stream;
            }

            void open() {
                compressionStream = FL2_createCStream();
                FL2_initCStream(compressionStream, 9);
            }

            void close() {
                FL2_outBuffer compressedBuffer = {cBuffer, internalBufferSize, 0};

                // Flush internal LZMA buffers
                unsigned int status;
                do {
                    status = FL2_endStream(compressionStream, &compressedBuffer);
                    fileStream->write((char*) compressedBuffer.dst, compressedBuffer.pos);
                    totalCompressedSize += compressedBuffer.pos;
                    compressedBuffer.pos = 0;
                } while (status);

                FL2_freeCStream(compressionStream);
            }

            size_t getTotalWrittenCompressedBytes() {
                return totalCompressedSize;
            }

            void write(std::array<DataType, internalBufferCount> &buffer, size_t itemCount) {
                assert(itemCount <= internalBufferCount);

                FL2_inBuffer uncompressedBuffer = {buffer.data(), itemCount * sizeof(DataType), 0};
                FL2_outBuffer compressedBuffer = {cBuffer, internalBufferSize, 0};

                // It may be necessary to write multiple buffers of data based on a single buffer worth of input data
                // Not great compression in that case, but it can happen
                do {
                    compressedBuffer.pos = 0;
                    FL2_compressStream(compressionStream, &compressedBuffer, &uncompressedBuffer);
                    fileStream->write((char*) compressedBuffer.dst, compressedBuffer.pos);
                    totalCompressedSize += compressedBuffer.pos;
                } while (compressedBuffer.pos == internalBufferSize);
            }
        };
    }
}