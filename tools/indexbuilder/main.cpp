#include <arrrgh.hpp>
#include <spinImage/cpu/index/types/Index.h>
#include <spinImage/cpu/index/IndexBuilder.h>

int main(int argc, const char** argv) {
    arrrgh::parser parser("indexbuilder", "Create indexes for QUICCI images.");
    const auto& indexDirectory = parser.add<std::string>(
            "index-directory", "The directory where the index should be stored.", '\0', arrrgh::Required, "");
    const auto& sourceDirectory = parser.add<std::string>(
            "quicci-dump-directory", "The directory where binary dump files of QUICCI images are stored that should be indexed.", '\0', arrrgh::Required, "");
    const auto& jsonDumpFile = parser.add<std::string>(
            "runtime-json-file", "Dump time measurement and statistics into the specified JSON file.", '\0', arrrgh::Optional, "/none/selected");
    const auto& appendToIndex = parser.add<bool>(
            "append", "Append to existing index.", '\0', arrrgh::Optional, false);
    const auto& fileStartIndex = parser.add<size_t>(
            "from-file-index", "Start adding objects at this index in the file listing.", '\0', arrrgh::Optional, 0);
    const auto& fileEndIndex = parser.add<size_t>(
            "to-file-index", "Add objects up to this index in the file listing.", '\0', arrrgh::Optional, 0);
    const auto& cacheNodeLimit = parser.add<size_t>(
            "cache-node-limit", "Sets the cache's node capacity.", '\0', arrrgh::Optional, 15000);
    const auto& imageLimit = parser.add<size_t>(
            "cache-image-limit", "Sets the cache's image capacity.", '\0', arrrgh::Optional, 500000);
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

    std::experimental::filesystem::path outJsonPath = jsonDumpFile.value();

    std::cout << "Building index from files in " << sourceDirectory.value() << "..." << std::endl;
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    SpinImage::index::build(sourceDirectory.value(), indexDirectory.value(),
            cacheNodeLimit.value(), imageLimit.value(), fileStartIndex.value(), fileEndIndex.value(), appendToIndex.value(), outJsonPath);

    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << std::endl << "Index construction complete. " << std::endl;
    std::cout << "Total execution time: " << float(duration.count()) / 1000.0f << " seconds" << std::endl;

    std::cout << std::endl << "Done." << std::endl;
}