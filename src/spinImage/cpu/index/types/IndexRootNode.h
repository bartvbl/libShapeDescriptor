#pragma once

#include <vector>
#include <iostream>

// The type of IndexNodeID is unsigned, so this expression
// represents the largest value possible, which we reserve for marking a link as disabled
const IndexNodeID ROOT_NODE_LINK_DISABLED = -1;

struct IndexRootNode {
    //std::vector<unsigned short> images;
    std::vector<IndexNodeID> links;
    // 1 bit per image/link. 0 = index node, 1 = bucket node
    std::vector<bool> linkTypes;

    IndexRootNode() {
        // Since the root node will most likely be mostly populated,
        // lookup speeds are massively increased by precreating all image entries,
        // and creating these entries only costs ~650k, it's worth the effort.
        //images.resize(65536);
        links.resize(65536);
        linkTypes.resize(65536);

        for(int i = 0; i < 65536; i++) {
            //images.at(i) = i;
            links.at(i) = ROOT_NODE_LINK_DISABLED;
            linkTypes.at(i) = INDEX_LINK_INDEX_NODE;
        }
    }
};