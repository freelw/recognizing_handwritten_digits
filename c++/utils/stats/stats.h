#ifndef AUTOGRAD_STATS_H
#define AUTOGRAD_STATS_H

#include <iostream>
namespace autograd {
    struct TmpMatricsStats {
        uint size;
        uint bytes;
    };
    struct TmpNodesStats {
        uint size;
        uint bytes;
    };
    struct TmpEdgesStats {
        uint size;
        uint bytes;
    };
    std::string stats();
} // namespace autograd
#endif