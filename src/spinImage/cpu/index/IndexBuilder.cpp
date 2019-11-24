#include <spinImage/utilities/fileutils.h>
#include <iostream>
#include "IndexBuilder.h"

Index SpinImage::index::build(std::string quicciImageDumpDirectory, std::string indexDumpDirectory) {
    std::vector<std::experimental::filesystem::path> filesInDirectory = SpinImage::utilities::listDirectory(quicciImageDumpDirectory);

    for(auto &path : filesInDirectory) {
        std::cout << path.filename() << std::endl;
    }
}