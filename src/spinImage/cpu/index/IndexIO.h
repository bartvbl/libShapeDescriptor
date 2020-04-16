#pragma once

#include <spinImage/cpu/index/types/Index.h>
#include <string>
#include <fstream>


namespace SpinImage {
    namespace index {
        namespace io {
            Index readIndex(std::experimental::filesystem::path indexDirectory);

            void writeIndex(const Index& index, std::experimental::filesystem::path indexDirectory);
        }
    }
}