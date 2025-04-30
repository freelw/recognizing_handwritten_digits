#include "tensor.h"
#include "backends/backend_ops.h"
#include "graph/node.h"
#include <iostream>

BackendOps *g_backend_ops = nullptr;

void test_plan() {
    std::cout << " print 1 " << std::endl;
    printAllTensors();
    Tensor *t = allocTensor({2, 2}, "t");
    Tensor *t1 = allocTensor({2}, "t1");
    graph::Node *node = graph::allocNode(t);
    graph::Node *node1 = graph::allocNode(t1);
    auto n = node->expand_add(node1);
    std::cout << " print 2 " << std::endl;
    printAllTensors();
    n->backward();
    std::cout << " print 3 " << std::endl;
    printAllTensors();
}

int main() {
    

    test_plan();

    freeAllTensors();
    freeAllTensorViews();
    graph::freeAllEdges();
    graph::freeAllNodes();
    freeAllActions();
    return 0;
}