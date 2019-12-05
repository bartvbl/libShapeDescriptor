#include <spinImage/utilities/fileutils.h>
#include <iostream>
#include <ZipLib/ZipArchive.h>
#include <ZipLib/ZipFile.h>
#include <spinImage/cpu/types/QUICCIImages.h>
#include <spinImage/utilities/readers/quicciReader.h>
#include <spinImage/cpu/index/types/MipmapStack.h>
#include "IndexBuilder.h"

Index SpinImage::index::build(std::string quicciImageDumpDirectory, std::string indexDumpDirectory) {
    std::vector<std::experimental::filesystem::path> filesInDirectory = SpinImage::utilities::listDirectory(quicciImageDumpDirectory);

    Index index;

    const unsigned int uintsPerQUICCImage = (spinImageWidthPixels * spinImageWidthPixels) / 32;
    unsigned int fileIndex = 0;
    for(auto &path : filesInDirectory) {
        fileIndex++;
        std::string archivePath = path.string();
        std::cout << "Adding file " << fileIndex << "/" << filesInDirectory.size() << ": " << archivePath << std::endl;
        index.indexedFileList.emplace_back(archivePath);


        SpinImage::cpu::QUICCIImages images = SpinImage::read::QUICCImagesFromDumpFile(archivePath);

        std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
#pragma omp parallel for
        for(size_t image = 0; image < images.imageCount; image++) {
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



    return index;
}