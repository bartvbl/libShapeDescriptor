#include <iomanip>
#include <fstream>
#include <cassert>
#include "IndexIO.h"

std::string formatFileIndex(IndexNodeID nodeID, const unsigned int nodes_per_file) {
    IndexNodeID fileID = (nodeID / nodes_per_file) + 1;
    std::stringstream ss;
    ss << std::setw(10) << std::setfill('0') << fileID;
    return ss.str();
}

std::string formatEntryIndex(IndexNodeID nodeID, const unsigned int nodes_per_file) {
    std::stringstream ss;
    ss << std::setw(10) << std::setfill('0') << nodeID;
    return ss.str();
}

IndexNode *index::io::readIndexNode(const std::experimental::filesystem::path& indexRootDirectory, IndexNodeID nodeID, const unsigned int fileGroupSize) {
    IndexNode* indexNode = new IndexNode(nodeID);

    std::experimental::filesystem::path indexFilePath = indexRootDirectory / "nodes" / (formatFileIndex(nodeID, fileGroupSize) + ".idx");

    // One extra 0 because of 0-termination
    std::array<char, 5> headerTitle = {0, 0, 0, 0, 0};
    IndexNodeID headerNodeID;
    unsigned long headerLinkArrayLength;
    unsigned long headerImageArrayLength;

    return indexNode;
}

void index::io::writeIndexNodes(const std::experimental::filesystem::path& indexRootDirectory, const std::vector<IndexNode *> &nodes, const unsigned int fileGroupSize) {
    std::experimental::filesystem::path indexDirectory = indexRootDirectory / "nodes";
    std::experimental::filesystem::create_directories(indexDirectory);

    std::experimental::filesystem::path indexFile = indexDirectory / (formatFileIndex(nodes.at(0)->id, fileGroupSize) + ".idx");

    for(unsigned int entryIndex = 0; entryIndex < nodes.size(); entryIndex++) {

        const std::string filename = "index_node_" + formatEntryIndex(nodes.at(entryIndex)->id, fileGroupSize) + ".dat";

        std::basic_stringstream<char> outStream;

        //unsigned long linkSize = node->links.size();
        //unsigned long imageSize = node->images.size();

        /*outStream << "INDX";
        outStream.write((char *) &node->id, sizeof(IndexNodeID));
        outStream.write((char *) &linkSize, sizeof(unsigned long));
        outStream.write((char *) &imageSize, sizeof(unsigned long));
        outStream.write((char *) node->links.data(), linkSize * sizeof(IndexNodeID));
        outStream.write((char *) node->linkTypes.data(), node->linkTypes.sizeInBytes());
        outStream.write((char *) node->images.data(), imageSize * sizeof(unsigned int));

        auto entry = archive->CreateEntry(filename);
        entry->UseDataDescriptor(); // read stream only once
        entry->SetCompressionStream(outStream,
                                    DeflateMethod::Create(),
                                    ZipArchiveEntry::CompressionMode::Immediate);*/
    }

    //ZipFile::SaveAndClose(archive, indexFile.string());
}

BucketNode *index::io::readBucketNode(const std::experimental::filesystem::path& indexRootDirectory, IndexNodeID nodeID, const unsigned int fileGroupSize) {
    BucketNode* bucketNode = new BucketNode(nodeID);

    std::experimental::filesystem::path indexFilePath = indexRootDirectory / "buckets" / (formatFileIndex(nodeID, fileGroupSize) + ".bkt");

    std::array<char, 5> headerTitle = {0, 0, 0, 0, 0};
    IndexNodeID headerNodeID;
    unsigned long headerIndexEntryCount;

    assert(std::string(headerTitle.data()) == "BCKT");

    bucketNode->images.resize(headerIndexEntryCount);

    return bucketNode;
}

void index::io::writeBucketNodes(const std::experimental::filesystem::path& indexRootDirectory, const std::vector<BucketNode *> &nodes, const unsigned int fileGroupSize) {
    std::experimental::filesystem::path bucketDirectory = indexRootDirectory / "buckets";
    std::experimental::filesystem::create_directories(bucketDirectory);

    std::experimental::filesystem::path bucketFile = bucketDirectory / (formatFileIndex(nodes.at(0)->id, fileGroupSize) + ".bkt");


    for(unsigned int entryIndex = 0; entryIndex < nodes.size(); entryIndex++) {

        std::basic_stringstream<char> outStream;

        const std::string filename = "bucket_node_" + formatEntryIndex(nodes.at(entryIndex)->id, fileGroupSize) + ".dat";

        BucketNode* node = nodes.at(entryIndex);

        unsigned long imageSize = node->images.size();

        outStream << "BCKT";
        outStream.write((char *) &node->id, sizeof(IndexNodeID));
        outStream.write((char *) &imageSize, sizeof(unsigned long));
        outStream.write((char *) node->images.data(), imageSize * sizeof(IndexEntry));

        //auto entry = archive->CreateEntry(filename);
        //entry->UseDataDescriptor(); // read stream only once
        //entry->SetCompressionStream(outStream,
        //                            DeflateMethod::Create(),
        //                            ZipArchiveEntry::CompressionMode::Immediate);
    }
}





Index index::io::loadIndex(std::experimental::filesystem::path rootFile) {
    return Index(std::experimental::filesystem::path(), nullptr, 0, 0);
}

void index::io::writeIndex(Index index, std::experimental::filesystem::path outDirectory) {

}
