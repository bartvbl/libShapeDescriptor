#pragma once

#include <vector>

// vector<bool> is indeed a thing, but you can't directly serialise it
// therefore, this library contains its own implementation that _is_.

class BoolVector {
private:
    unsigned long length = 0;
    std::vector<unsigned int> container;
public:
    void push_back(bool bit);
    bool at(unsigned long index);
    void set(unsigned long index, bool value);
    void reserve(unsigned long size);
    void resize(unsigned long size);
    unsigned long size();
    unsigned char* data();
    unsigned long sizeInBytes();
};