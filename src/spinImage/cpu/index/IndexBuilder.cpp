#include <spinImage/utilities/fileutils.h>
#include <iostream>
#include <spinImage/cpu/types/QUICCIImages.h>
#include <spinImage/utilities/readers/quicciReader.h>
#include <spinImage/cpu/index/types/MipmapStack.h>
#include <bitset>
#include "IndexBuilder.h"
#include "NodeBlockCache.h"


const unsigned int uintsPerMipmapImageLevel[4] = {0, 2, 8, 32};

bool isImagePairEquivalent(const unsigned int* image1, const unsigned int* image2, const unsigned int level) {
    for(unsigned int i = 0; i < uintsPerMipmapImageLevel[level]; i++) {
        if(image1[i] != image2[i]) {
            return false;
        }
    }
    return true;
}

Index SpinImage::index::build(std::string quicciImageDumpDirectory, std::string indexDumpDirectory) {
    std::vector<std::experimental::filesystem::path> filesInDirectory = SpinImage::utilities::listDirectory(quicciImageDumpDirectory);
    std::experimental::filesystem::path indexDirectory(indexDumpDirectory);

    std::vector<std::experimental::filesystem::path>* indexedFiles =
            new std::vector<std::experimental::filesystem::path>();
    indexedFiles->reserve(filesInDirectory.size());

    NodeBlock rootBlock;

    // Size requirements:
    // - Per node: 64 bytes + (136 x stored image count)
    // - Per node block: 256 x node size
    // - Cache total: 65536 x node block = 1GB + total size of stored images
    NodeBlockCache cache(65536, indexDirectory, &rootBlock);


    const unsigned int uintsPerQUICCImage = (spinImageWidthPixels * spinImageWidthPixels) / 32;
    IndexFileID fileIndex = 0;
    for(const auto &path : filesInDirectory) {
        const std::string archivePath = path.string();
        std::cout << "Adding file " << (fileIndex + 1) << "/" << filesInDirectory.size() << ": " << archivePath << std::endl;
        indexedFiles->emplace_back(archivePath);

        SpinImage::cpu::QUICCIImages images = SpinImage::read::QUICCImagesFromDumpFile(archivePath);


        std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

        for (IndexImageID imageIndex = 0; imageIndex < images.imageCount; imageIndex++) {
            MipmapStack combined = MipmapStack::combine(
                    images.horizontallyIncreasingImages + imageIndex * uintsPerQUICCImage,
                    images.horizontallyDecreasingImages + imageIndex * uintsPerQUICCImage);
            IndexEntry entry = {fileIndex, imageIndex};
            cache.insertImage(combined, entry);
        }

        std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        std::cout << "\tTook " << float(duration.count()) / 1000.0f << " seconds." << std::endl;

        delete[] images.horizontallyIncreasingImages;
        delete[] images.horizontallyDecreasingImages;

        fileIndex++;
    }

    // Ensuring all changes are written to disk
    //cache.flush();

    // Final construction of the index
    Index index(indexDirectory, indexedFiles, rootBlock, 0, 0);

    return index;
}