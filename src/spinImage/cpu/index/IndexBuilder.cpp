#include <spinImage/utilities/fileutils.h>
#include <iostream>
#include <spinImage/cpu/types/QUICCIImages.h>
#include <spinImage/utilities/readers/quicciReader.h>
#include <spinImage/cpu/index/types/MipmapStack.h>
#include <bitset>
#include "IndexBuilder.h"
#include "NodeBlockCache.h"

Index SpinImage::index::build(std::string quicciImageDumpDirectory, std::string indexDumpDirectory) {
    std::vector<std::experimental::filesystem::path> filesInDirectory = SpinImage::utilities::listDirectory(quicciImageDumpDirectory);
    std::experimental::filesystem::path indexDirectory(indexDumpDirectory);

    std::vector<std::experimental::filesystem::path>* indexedFiles =
            new std::vector<std::experimental::filesystem::path>();
    indexedFiles->reserve(filesInDirectory.size());

    NodeBlock rootBlock;

    NodeBlockCache cache(20000, indexDirectory, &rootBlock);

    const unsigned int uintsPerQUICCImage = (spinImageWidthPixels * spinImageWidthPixels) / 32;
#pragma omp parallel for
    for(unsigned int fileIndex = 0; fileIndex < filesInDirectory.size(); fileIndex++) {
        std::experimental::filesystem::path path = filesInDirectory.at(fileIndex);
        const std::string archivePath = path.string();
#pragma omp critical
        std::cout << "Adding file " << (fileIndex + 1) << "/" << filesInDirectory.size() << ": " << archivePath << std::endl;
#pragma omp critical
        indexedFiles->emplace_back(archivePath);

        SpinImage::cpu::QUICCIImages images = SpinImage::read::QUICCImagesFromDumpFile(archivePath);

        std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

        for (IndexImageID imageIndex = 0; imageIndex < images.imageCount; imageIndex++) {
            MipmapStack combined = MipmapStack::combine(
                    images.horizontallyIncreasingImages + imageIndex * uintsPerQUICCImage,
                    images.horizontallyDecreasingImages + imageIndex * uintsPerQUICCImage);
            IndexEntry entry = {fileIndex, imageIndex};
#pragma omp critical
            cache.insertImage(combined, entry);
        }

        std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        std::cout << "\tTook " << float(duration.count()) / 1000.0f << " seconds." << std::endl;

        delete[] images.horizontallyIncreasingImages;
        delete[] images.horizontallyDecreasingImages;
    }

    // Ensuring all changes are written to disk
    std::cout << "Flushing cache.." << std::endl;
    cache.flush();

    // Final construction of the index
    Index index(indexDirectory, indexedFiles, rootBlock);

    // Write the root node to disk
    std::cout << "Writing core index files.." << std::endl;
    SpinImage::index::io::writeNodeBlock(&rootBlock, indexDirectory);
    SpinImage::index::io::writeIndex(index, indexDirectory);

    return index;
}