#include <spinImage/utilities/fileutils.h>
#include <iostream>
#include <ZipLib/ZipArchive.h>
#include <ZipLib/ZipFile.h>
#include <spinImage/cpu/types/QUICCIImages.h>
#include "IndexBuilder.h"

Index SpinImage::index::build(std::string quicciImageDumpDirectory, std::string indexDumpDirectory) {
    std::vector<std::experimental::filesystem::path> filesInDirectory = SpinImage::utilities::listDirectory(quicciImageDumpDirectory);

    unsigned int fileIndex = 0;
    for(auto &path : filesInDirectory) {
        fileIndex++;
        std::string archivePath = path.string();
        std::cout << "Adding file " << fileIndex << "/" << filesInDirectory.size() << ": " << archivePath << std::endl;
        ZipArchive::Ptr archive = ZipFile::Open(archivePath);

        ZipArchiveEntry::Ptr entry = archive->GetEntry("quicci_images.dat");
        std::istream* decompressStream = entry->GetDecompressionStream();

        char header[4];
        size_t imageCount;
        unsigned int descriptorWidthPixels;

        decompressStream->read(header, 4);
        decompressStream->read((char*) &imageCount, sizeof(size_t));
        decompressStream->read((char*) &descriptorWidthPixels, sizeof(unsigned int));

        if(std::string(header) != "QUIC") {
            std::cout << "WARNING: File header does not match expectations, and is thus possibly corrupt." << std::endl;
        }

        std::cout << "\tFile has " << imageCount << " images" << std::endl;
        if(descriptorWidthPixels != spinImageWidthPixels) {
            std::cout << "The libSpinImage library was compiled with a different image size compared to those stored in this file." << std::endl;
            std::cout << "This means any processing this program does on them will not work correctly." << std::endl;
            std::cout << "Skipping this file.." << std::endl;
            continue;
        }

        const unsigned int uintsPerQUICCImage = (spinImageWidthPixels * spinImageWidthPixels) / 32;
        SpinImage::cpu::QUICCIImages images;
        images.imageCount = imageCount;
        images.horizontallyIncreasingImages = new unsigned int[uintsPerQUICCImage * imageCount];
        images.horizontallyDecreasingImages = new unsigned int[uintsPerQUICCImage * imageCount];

        decompressStream->read((char*) images.horizontallyIncreasingImages, uintsPerQUICCImage * imageCount * sizeof(unsigned int));
        decompressStream->read((char*) images.horizontallyDecreasingImages, uintsPerQUICCImage * imageCount * sizeof(unsigned int));

        for(size_t image = 0; image < imageCount; image++) {

        }

        delete[] images.horizontallyIncreasingImages;
        delete[] images.horizontallyDecreasingImages;
    }

    Index index;

    return index;
}