//
// Created by bart on 22.11.19.
//

#include "IndexIO.h"

IndexNodeID getIndexNodeCount(std::experimental::filesystem::path indexRootDirectory) {
    std::experimental::filesystem::path indexNodeFolder = indexRootDirectory / "nodes";
    return 0;

}

IndexNodeID getBucketNodeCount(std::experimental::filesystem::path indexRootDirectory) {
    std::experimental::filesystem::path bucketFolder = indexRootDirectory / "buckets";
    return 0;
}
