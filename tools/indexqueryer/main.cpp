#include <arrrgh.hpp>
#include <spinImage/cpu/index/types/Index.h>
#include <spinImage/cpu/index/IndexIO.h>
#include <spinImage/cpu/index/IndexQueryer.h>
#include <lodepng.h>

int main(int argc, const char** argv) {
    arrrgh::parser parser("queryindex", "Query an existing index of QUICCI images.");
    const auto& indexDirectory = parser.add<std::string>(
            "index-directory", "The location of the directory containing the existing index.", '\0', arrrgh::Required, "");
    const auto& queryImage = parser.add<std::string>(
            "query-image-file", "The location of the PNG file representing the image that should be queried.", '\0', arrrgh::Required, "");
    const auto& showHelp = parser.add<bool>(
            "help", "Show this help message.", 'h', arrrgh::Optional, false);

    try
    {
        parser.parse(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        parser.show_usage(std::cerr);
        exit(1);
    }

    // Show help if desired
    if(showHelp.value())
    {
        return 0;
    }

    std::cout << "Reading query image.." << std::endl;
    std::experimental::filesystem::path imageFilePath = queryImage.value();
    if(!std::experimental::filesystem::exists(imageFilePath)) {
        std::cerr << "Query image file " << std::experimental::filesystem::absolute(imageFilePath) << " was not found." << std::endl;
        return 0;
    }

    std::vector<unsigned char> imageData;
    unsigned int width;
    unsigned int height;
    lodepng::decode(imageData, width, height, imageFilePath, LCT_RGBA);
    std::cout << "Image is " << width << "x" << height << std::endl;

    assert(width == spinImageWidthPixels && height == spinImageWidthPixels);

    unsigned int* queryQUIICIMage = new unsigned int[(spinImageWidthPixels * spinImageWidthPixels) / 32];

    for(int row = 0; row < spinImageWidthPixels; row++) {
        std::bitset<32> bitQueue(0);
        for(int col = 0; col < spinImageWidthPixels; col++) {
            bitQueue[col % 32] = imageData.at(4 * (row * spinImageWidthPixels + col)) != 0;
            if(col % 32 == 0) {
                queryQUIICIMage[row * (spinImageWidthPixels / 32) + (col / 32)] = bitQueue.to_ulong();
                bitQueue.reset();
            }
        }
    }

    std::cout << "Reading index metadata.." << std::endl;
    Index index = SpinImage::index::io::readIndex(indexDirectory.value());

    std::vector<IndexEntry> searchResults = queryIndex(index, queryQUIICIMage, 100);


    delete[] queryQUIICIMage;

}