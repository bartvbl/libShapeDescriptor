#include <spinImage/utilities/fileutils.h>
#include <iostream>
#include <ZipLib/ZipArchive.h>
#include <ZipLib/ZipFile.h>
#include <spinImage/cpu/types/QUICCIImages.h>
#include <spinImage/utilities/readers/quicciReader.h>
#include <spinImage/cpu/index/types/MipmapStack.h>
#include "IndexBuilder.h"
#include "IndexFileCache.h"

Index SpinImage::index::build(std::string quicciImageDumpDirectory, std::string indexDumpDirectory) {
    std::vector<std::experimental::filesystem::path> filesInDirectory = SpinImage::utilities::listDirectory(quicciImageDumpDirectory);
    std::experimental::filesystem::path indexDirectory(indexDumpDirectory);

    // The index node capacity is set quite high to allow most of the index to be in memory during construction
    IndexFileCache cache(indexDirectory, 1000, 1000);

    std::vector<std::experimental::filesystem::path>* indexedFiles = new std::vector<std::experimental::filesystem::path>();
    indexedFiles->reserve(5000);

    const unsigned int uintsPerQUICCImage = (spinImageWidthPixels * spinImageWidthPixels) / 32;
    IndexFileID fileIndex = 0;
    for(const auto &path : filesInDirectory) {
        fileIndex++;
        const std::string archivePath = path.string();
        std::cout << "Adding file " << fileIndex << "/" << filesInDirectory.size() << ": " << archivePath << std::endl;
        indexedFiles->emplace_back(archivePath);

        SpinImage::cpu::QUICCIImages images = SpinImage::read::QUICCImagesFromDumpFile(archivePath);



        std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
#pragma omp parallel for
        for(IndexImageID image = 0; image < images.imageCount; image++) {
            MipmapStack mipmapsIncreasing(images.horizontallyIncreasingImages + image * uintsPerQUICCImage);
            MipmapStack mipmapsDecreasing(images.horizontallyDecreasingImages + image * uintsPerQUICCImage);



            // Follow index until mipmap is not found in index node
            // If the node you end up in is a bucket node, add the node to it
                // If the bucket node is full, and it's not at the deepest level, split the node and divide its
                // contents over its children. Change node to an index node
            // If the trail goes cold at an index node, the index node has already been split,
            // and needs its own separate bucket node. Create it, add it to the index, and add the image to it.
        }

        std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        std::cout << "\tTook " << float(duration.count()) / 1000.0f << " seconds." << std::endl;

        delete[] images.horizontallyIncreasingImages;
        delete[] images.horizontallyDecreasingImages;
    }

    // Ensuring all changes are written to disk
    cache.flush();

    // Copying the data into data structures that persist after the function exits
    IndexRootNode* duplicatedRootNode = new IndexRootNode();
    *duplicatedRootNode = cache.rootNode;

    // Final construction of the index
    Index index(indexDirectory, indexedFiles, duplicatedRootNode);

    return index;
}