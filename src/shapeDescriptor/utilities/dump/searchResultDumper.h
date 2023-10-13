#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include <tsl/ordered_map.h>
#include <json.hpp>
#include <fstream>

namespace ShapeDescriptor {
    namespace dump {
        using json = nlohmann::ordered_json;

        template<typename ScoreType>
        void searchResults(ShapeDescriptor::cpu::array<SearchResults<ScoreType>> searchResults, std::string outputFilePath) {
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
