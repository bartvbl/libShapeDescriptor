#include <catch2/catch_test_macros.hpp>
#include <shapeDescriptor/shapeDescriptor.h>
/*TEST_CASE("Reading and writing of compressed files", "[index]" ) {

    SECTION("Compression") {
        // 128 * 4MB
        const unsigned int count = 1024*1024*128;
        unsigned int* arbitraryDataBuffer = new unsigned int[count];

        for(unsigned int i = 0; i < count; i++) {
            arbitraryDataBuffer[i] = i;
        }

        std::experimental::filesystem::path tempFile =
                std::experimental::filesystem::temp_directory_path() / "test_index.dat";

        SpinImage::utilities::writeCompressedFile(reinterpret_cast<const char *>(arbitraryDataBuffer), count * sizeof(unsigned int), tempFile);

        size_t fileSize;
        SpinImage::utilities::readCompressedFile()

        delete[] arbitraryDataBuffer;
    }

    SECTION("float3 length") {
        SpinImage::cpu::float3 vertex;
        vertex.x = 1;
        vertex.y = 0;
        vertex.z = 0;
        REQUIRE(length(vertex) == 1);
    }
}*/