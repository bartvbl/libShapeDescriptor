#pragma once

#include <string>
#include "NodeBlock.h"
#include <spinImage/utilities/Cache.h>
template<typename FodderType>
class CannonFodderCache : public Cache<std::string, FodderType> {
protected:
    void eject(NodeBlock* item) override;
    void onEviction(NodeBlock* item) override;
    NodeBlock* load(std::string &itemID) override;

public:
    CannonFodderCache(size_t nodeBlockCapacity) : Cache<std::string, FodderType>(nodeBlockCapacity) {}
    void insertSomeBlock(size_t ID, FodderType* item);
};

template<typename FodderType>
void CannonFodderCache<FodderType>::eject(NodeBlock *item) {
    // Don't care
}

template<typename FodderType>
void CannonFodderCache<FodderType>::onEviction(NodeBlock *item) {
    // Don't care
}

template<typename FodderType>
NodeBlock *CannonFodderCache<FodderType>::load(std::string &itemID) {
    return new NodeBlock();
}

template<typename FodderType>
void CannonFodderCache<FodderType>::insertSomeBlock(size_t ID, FodderType* item) {
    std::string idString = std::to_string(ID);
    this->insertItem(idString, item, true, false);
}
