#include <arrrgh.hpp>

int main(int argc, char** argv) {
    arrrgh::parser parser("indexer", "Create and use indexes for QUICCI images.");
    const auto& indexFile = parser.add<std::string>(
            "index-file", "The location of the input OBJ model file.", '\0', arrrgh::Required, "");
    const auto& showHelp = parser.add<bool>(
            "help", "Show this help message.", 'h', arrrgh::Optional, false);

}