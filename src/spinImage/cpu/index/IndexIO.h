#pragma once

#include <spinImage/cpu/index/types/Index.h>
#include <string>

namespace index {
    namespace io {
        Index loadIndex(std::experimental::filesystem::path rootFile);
        void writeIndex(Index index, std::experimental::filesystem::path outDirectory);


    }
}

