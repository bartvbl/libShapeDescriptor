#include <spinImage/libraryBuildSettings.h>
#include <sstream>
#include "rawDescriptorDumper.h"

void SpinImage::dump::raw::descriptors(
        const std::string &outputDumpFile,
        const SpinImage::cpu::QUICCIImages &images) {
    const size_t bytesPerQUICCImage = ((spinImageWidthPixels * spinImageWidthPixels) / 32) * sizeof(unsigned int);
    const unsigned int imageWidthPixels = spinImageWidthPixels;

    std::basic_stringstream<char> outStream;

    outStream << "QUIC";
    outStream.write((char*) &images.imageCount, sizeof(size_t));
    outStream.write((char*) &imageWidthPixels, sizeof(unsigned int));
    outStream.write((char*) images.horizontallyIncreasingImages, images.imageCount * bytesPerQUICCImage);
    outStream.write((char*) images.horizontallyDecreasingImages, images.imageCount * bytesPerQUICCImage);

    //auto archive = ZipFile::Open(outputDumpFile);
    //auto entry = archive->CreateEntry("quicci_images.dat");
    //entry->UseDataDescriptor(); // read stream only once
    //entry->SetCompressionStream(outStream);
    //ZipFile::SaveAndClose(archive, outputDumpFile);
}