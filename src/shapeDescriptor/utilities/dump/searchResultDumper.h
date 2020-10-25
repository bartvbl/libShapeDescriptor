#pragma once

#include <shapeDescriptor/libraryBuildSettings.h>
#include <shapeDescriptor/gpu/spinImageSearcher.cuh>
#include <tsl/ordered_map.h>
#include <json.hpp>
#include <fstream>

namespace ShapeDescriptor {
    namespace dump {
        template<class Key, class T, class Ignore, class Allocator,
                class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>,
                class AllocatorPair = typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, T>>,
                class ValueTypeContainer = std::vector<std::pair<Key, T>, AllocatorPair>>
        using ordered_map = tsl::ordered_map<Key, T, Hash, KeyEqual, AllocatorPair, ValueTypeContainer>;

        using json = nlohmann::basic_json<ordered_map>;

        template<typename ScoreType>
        void searchResults(ShapeDescriptor::cpu::array<gpu::SearchResults<ScoreType>> searchResults, std::string outputFilePath) {
            json outJson;

            outJson["version"] = "v1";
            outJson["queryCount"] = searchResults.length;
            outJson["searchResultsPerQuery"] = SEARCH_RESULT_COUNT;
            outJson["results"] = {};

            for(size_t image = 0; image < searchResults.length; image++) {
                outJson["results"].emplace_back();
                outJson["results"][image] = {};

                outJson["results"][image]["scores"] = {};
                for (unsigned int i = 0; i < SEARCH_RESULT_COUNT; i++) {
                    outJson["results"][image]["scores"].push_back(searchResults.content[image].scores[i]);
                }

                outJson["results"][image]["indices"] = {};
                for (unsigned int i = 0; i < SEARCH_RESULT_COUNT; i++) {
                    outJson["results"][image]["indices"].push_back(searchResults.content[image].indices[i]);
                }
            }

            std::ofstream outFile(outputFilePath);
            outFile << outJson.dump(4);
            outFile.close();
        }
    }
}
