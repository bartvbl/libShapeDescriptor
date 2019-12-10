#include <ZipLib/ZipFile.h>
#include <iomanip>
#include "IndexIO.h"

std::string formatFileIndex(IndexNodeID nodeID) {
    std::stringstream ss;
    ss << std::setw(10) << std::setfill('0') << nodeID;
    return ss.str();
}

IndexNode *index::io::readIndexNode(std::experimental::filesystem::path indexRootDirectory, IndexNodeID nodeID) {
    return nullptr;
}

void index::io::writeIndexNode(std::experimental::filesystem::path indexRootDirectory, IndexNode *node) {
    std::basic_stringstream<char> outStream;

    outStream << "INDX";
    outStream.write((char*) node->id, sizeof(IndexNodeID));
    outStream.write((char*) node->images.size(), sizeof(unsigned long));
    outStream.write((char*) node->links, node->images.size() * sizeof(unsigned int));
    outStream.write((char*) node->images.data(), node->images.size() * sizeof(unsigned int));


    std::experimental::filesystem::path indexDirectory = indexRootDirectory / "nodes";
    std::experimental::filesystem::create_directories(indexDirectory);

    std::experimental::filesystem::path indexFile = indexDirectory / (formatFileIndex(node->id) + ".idx");

    auto archive = ZipFile::Open(indexFile.string());
    auto entry = archive->CreateEntry("index_node.dat");
    entry->UseDataDescriptor(); // read stream only once
    entry->SetCompressionStream(outStream);
    ZipFile::SaveAndClose(archive, indexFile.string());
}

BucketNode *index::io::readBucketNode(std::experimental::filesystem::path indexRootDirectory, IndexNodeID nodeID) {
    return nullptr;
}

void index::io::writeBucketNode(std::experimental::filesystem::path indexRootDirectory, BucketNode *node) {
    std::basic_stringstream<char> outStream;

    outStream << "BCKT";
    outStream.write((char*) node->id, sizeof(IndexNodeID));
    outStream.write((char*) node->images.size(), sizeof(unsigned long));
    outStream.write((char*) node->images.data(), node->images.size() * sizeof(IndexEntry));

    std::experimental::filesystem::path bucketDirectory = indexRootDirectory / "buckets";
    std::experimental::filesystem::create_directories(bucketDirectory);

    std::experimental::filesystem::path bucketFile = bucketDirectory / (formatFileIndex(node->id) + ".bkt");

    auto archive = ZipFile::Open(bucketFile.string());
    auto entry = archive->CreateEntry("bucket_node.dat");
    entry->UseDataDescriptor(); // read stream only once
    entry->SetCompressionStream(outStream);
    ZipFile::SaveAndClose(archive, bucketFile.string());
}





Index index::io::loadIndex(std::experimental::filesystem::path rootFile) {
    return Index(std::experimental::filesystem::path(), nullptr, nullptr, 0, 0);
}

void index::io::writeIndex(Index index, std::experimental::filesystem::path outDirectory) {

}
