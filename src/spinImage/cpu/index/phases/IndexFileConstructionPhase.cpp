#include "IndexFileConstructionPhase.h"

// === File format ===
// Header:
//      - "QIDX": File identifier
//      - Array of 4096 pointers to all 1-pixel pattern nodes
//      - List of strings containing paths to the indexed data files
// File contents:
//      - List of pattern blocks
//          - n x 4 bytes: Bit list of existing child nodes.
//                         Order and length depends on pattern itself
//          - 4 + 4 bytes: Length of 'shortcuts' list.
//                         First 4 bytes: number of shortcuts
//                         Last 4 bytes: size of compressed list (bytes)
//          - n bytes:     Compressed block of shortcuts
//                         Shortcut format:
//                             - 512 bytes: Pattern Image
//                             - 8 bytes: Pointer to pattern block
//          - n bytes:     Compressed block of pattern entries
//                         List is sorted by remaining pixel count
//                         Pattern entry format:
//                             - 4 bytes: File index
//                             - 4 bytes: Image index within file
//                             - 2 bytes: Number of set pixels present in image but not part of this pattern