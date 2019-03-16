#pragma once

#include <cstddef>
#include <vector>
#include <map>

class Histogram {
private:
    std::map<unsigned int, size_t> contents;
    void ensureKeyExists(unsigned int key, std::map<unsigned int, size_t> &map) const;

public:
    void count(size_t key);
    Histogram merge(Histogram other);
    std::string toJSON();

};

