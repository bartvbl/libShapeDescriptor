#include <algorithm>
#include <sstream>
#include "Histogram.h"

void Histogram::ensureKeyExists(unsigned int key, std::map<unsigned int, size_t> &map) const {
    auto it = map.find(key);
    if(it == map.end()) {
        map[key] = 0;
    }
}

Histogram Histogram::merge(Histogram other) {
    Histogram mergedHistogram;

    // Dump elements from this histogram first
    for (auto &content : contents) {
        mergedHistogram.contents[content.first] = content.second;
    }

    for (auto &content : other.contents) {
        ensureKeyExists(content.first, mergedHistogram.contents);
        mergedHistogram.contents[content.first] += content.second;
    }

    return mergedHistogram;
}

void Histogram::count(size_t key) {
    ensureKeyExists(key, contents);
    contents[key]++;
}

std::string Histogram::toJSON() {
    std::vector<unsigned int> keys;
    for (auto &content : contents) {
        keys.push_back(content.first);
    }

    std::sort(keys.begin(), keys.end());

    std::stringstream ss;

    ss << "{" << std::endl;
    for (auto &key : keys) {
        ss << "\t\"" << key << "\": " << contents[key] << "," << std::endl;
    }
    ss << "}" << std::endl;

    return ss.str();
}
