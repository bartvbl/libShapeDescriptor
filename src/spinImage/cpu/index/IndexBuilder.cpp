#include <spinImage/utilities/fileutils.h>
#include <iostream>
#include <ZipLib/ZipArchive.h>
#include <ZipLib/ZipFile.h>
#include <spinImage/cpu/types/QUICCIImages.h>
#include <spinImage/utilities/readers/quicciReader.h>
#include <spinImage/cpu/index/types/MipmapStack.h>
#include "IndexBuilder.h"
#include "IndexFileCache.h"

const unsigned int uintsPerMipmapImageLevel[4] = {0, 2, 8, 32};

bool isImagePairEquivalent(const unsigned int* image1, const unsigned int* image2, const unsigned int level) {
    for(unsigned int i = 0; i < uintsPerMipmapImageLevel[level]; i++) {
        if(image1[i] != image2[i]) {
            return false;
        }
    }
    return true;
}

IndexNodeID processLink(IndexFileCache &cache, const unsigned int nextLink, const unsigned int* mipmapImage, const unsigned int level) {
    const IndexNode* indexNode = cache.fetchIndexNode(nextLink);
    for(int image = 0; image < indexNode->links.size(); image++) {
        const unsigned int* indexNodeImage = indexNode->images.data() + uintsPerMipmapImageLevel[level] * image;
        if(isImagePairEquivalent(indexNodeImage, mipmapImage, level)) {
            return indexNode->links.at(image);
        }
    }
    return cache.createIndexNode(nextLink, mipmapImage, level);
}

IndexNodeID processBucketLink(IndexFileCache &cache, const unsigned int nextLink, const unsigned int* mipmapImage, const unsigned int level) {
    const IndexNode* indexNode = cache.fetchIndexNode(nextLink);
    for(int image = 0; image < indexNode->links.size(); image++) {
        const unsigned int* indexNodeImage = indexNode->images.data() + uintsPerMipmapImageLevel[level] * image;
        if(isImagePairEquivalent(indexNodeImage, mipmapImage, level)) {
            return indexNode->links.at(image);
        }
    }
    return cache.createBucketNode(nextLink, mipmapImage, level);
}

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
//#pragma omp parallel for
        for(IndexImageID image = 0; image < images.imageCount; image++) {
            //MipmapStack mipmapsIncreasing(images.horizontallyIncreasingImages + image * uintsPerQUICCImage);
            //MipmapStack mipmapsDecreasing(images.horizontallyDecreasingImages + image * uintsPerQUICCImage);

            MipmapStack combined = MipmapStack::combine(
                    images.horizontallyIncreasingImages + image * uintsPerQUICCImage,
                    images.horizontallyDecreasingImages + image * uintsPerQUICCImage);

            IndexNodeID nextLink = cache.rootNode.links.at(combined.level0.image);
            if(nextLink == ROOT_NODE_LINK_DISABLED) {
                // Temporarily expand the 16-bit root image to avoid pointer cast shenanigans
                unsigned int rootMipmapImage = combined.level0.image;
                nextLink = cache.createIndexNode(0, &rootMipmapImage, 0);
            }

            // nextLink points to a valid index node at this point

            nextLink = processLink(cache, nextLink, combined.level1.image.data(), 1);
            nextLink = processLink(cache, nextLink, combined.level2.image.data(), 2);
            IndexNodeID bucketNodeID = processBucketLink(cache, nextLink, combined.level3.image.data(), 3);
            IndexEntry entry = {fileIndex-1, image};
            cache.insertImageIntoBucketNode(bucketNodeID, entry);
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
    Index index(indexDirectory, indexedFiles, duplicatedRootNode, 0, 0);

    return index;
}