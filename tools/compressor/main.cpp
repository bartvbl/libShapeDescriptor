#include <arrrgh.hpp>
#include <spinImage/cpu/index/types/Index.h>
#include <spinImage/cpu/index/IndexIO.h>
#include <spinImage/cpu/index/IndexQueryer.h>
#include <lodepng.h>
#include <fstream>
#include <spinImage/utilities/fileutils.h>

int main(int argc, const char** argv) {
    arrrgh::parser parser("compressor", "Compress and decompress files stored in the library's archive format.");
    const auto& inputFile = parser.add<std::string>(
            "input", "The input file to read from.", '\0', arrrgh::Required, "");
    const auto& outputFile = parser.add<std::string>(
            "output", "The output file to write to.", '\0', arrrgh::Required, "");
    const auto& compress = parser.add<bool>(
            "compress", "Compress the input file, and store it in the output file.", '\0', arrrgh::Optional, false);
    const auto& decompress = parser.add<bool>(
            "decompress", "Decompress the input file, and store it in the output file.", '\0', arrrgh::Optional, false);
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

    std::experimental::filesystem::path inputPath = inputFile.value();
    std::experimental::filesystem::path outputPath = outputFile.value();

    inputPath = std::experimental::filesystem::absolute(inputPath);
    outputPath = std::experimental::filesystem::absolute(outputPath);

    if(compress.value()) {
        size_t inputFileSize = std::experimental::filesystem::file_size(inputPath);
        char* inputFileContents = new char[inputFileSize];
        std::ifstream inputStream(inputPath, std::ios::out | std::ios::binary);
        inputStream.read(inputFileContents, inputFileSize);
        inputStream.close();
        SpinImage::utilities::writeCompressedFile(inputFileContents, inputFileSize, outputPath);
    } else if(decompress.value()) {
        size_t inputBufferSize = 0;
        const char* inputFileContents = SpinImage::utilities::readCompressedFile(inputPath, &inputBufferSize);
        std::fstream outStream = std::fstream(outputFile, std::ios::out | std::ios::binary);
        outStream.write(inputFileContents, inputBufferSize);
        outStream.close();
    }
}