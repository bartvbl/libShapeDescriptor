#pragma once

#include <spinImage/cpu/index/types/Index.h>
#include <string>

Index loadIndex(std::string rootFile);
void writeIndex(Index index, std::string outDirectory);