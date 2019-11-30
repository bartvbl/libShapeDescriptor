#include <ZipLib/ZipArchive.h>
#include <ZipLib/ZipFile.h>
#include <spinImage/libraryBuildSettings.h>
#include "quicciReader.h"
#include <exception>


SpinImage::cpu::QUICCIImages SpinImage::read::QUICCImagesFromDumpFile(std::string &dumpFileLocation) {
    ZipArchive::Ptr archive = ZipFile::Open(dumpFileLocation);

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
        std::cout << "Aborting index construction.." << std::endl;
        throw std::runtime_error("Invalid input file detected!");
    }

    const unsigned int uintsPerQUICCImage = (spinImageWidthPixels * spinImageWidthPixels) / 32;
    SpinImage::cpu::QUICCIImages images;
    images.imageCount = imageCount;
    images.horizontallyIncreasingImages = new unsigned int[uintsPerQUICCImage * imageCount];
    images.horizontallyDecreasingImages = new unsigned int[uintsPerQUICCImage * imageCount];

    decompressStream->read((char*) images.horizontallyIncreasingImages, uintsPerQUICCImage * imageCount * sizeof(unsigned int));
    decompressStream->read((char*) images.horizontallyDecreasingImages, uintsPerQUICCImage * imageCount * sizeof(unsigned int));

    return images;
}