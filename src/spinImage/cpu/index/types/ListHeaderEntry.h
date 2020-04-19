#pragma once

// We're interpreting raw bytes as this struct.
// Need to disable padding bytes for that to work
#pragma pack(push, 1)
struct ListHeaderEntry {
    unsigned short pixelCount;
    unsigned int imageReferenceCount;
};
#pragma pack(pop)