#include <iostream>
#include "layernorm.h"
#include "attention.h"

using namespace std;

void test_layernorm() {
    auto normalized_shape = 6;
    LayerNorm layernorm(normalized_shape);
    Matrix *input = allocTmpMatrix(Shape(normalized_shape, 2));
    for (int i = 0; i < normalized_shape; i++) {
        (*input)[i][0] = i;
        (*input)[i][1] = i+1;
    }
    std::vector<uint> labels = {2, 3};
    autograd::Node *x = autograd::allocNode(input);
    x->require_grad();
    autograd::Node *y = layernorm.forward(x);

    auto loss = y->CrossEntropy(labels);
    cout << "y: " << endl;
    cout << *y->get_weight() << endl;
    cout << "loss: " << endl;
    cout << *loss->get_weight() << endl;

    loss->backward();

    cout << "gamma beta grad: " << endl;
    for (auto param : layernorm.parameters()) {
        cout << *param->get_grad() << endl;
    }

    cout << "x grad: " << endl;
    cout << *x->get_grad() << endl;

    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
}

void test_softmax() {

    auto softmax_size = 6;
    Matrix *input = allocTmpMatrix(Shape(softmax_size, 2));
    for (int i = 0; i < 6; i++) {
        (*input)[i][0] = i;
        (*input)[i][1] = i+1;
    }
    std::vector<uint> labels = {2, 3};
    autograd::Node *x = autograd::allocNode(input);
    x->require_grad();

    autograd::Node *y = x->Softmax();

    auto loss = y->CrossEntropy(labels);

    loss->backward();

    // print y
    cout << "y: " << endl;
    cout << *y->get_weight() << endl;

    cout << "loss : " << endl;
    cout << *loss->get_weight() << endl;

    cout << "x grad: " << endl;
    cout << *x->get_grad() << endl;

    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
}

void test_attention(const std::vector<uint> &valid_lens) {

    std::vector<autograd::Node *> queries;

    Matrix *mq1 = allocTmpMatrix(Shape(2, 1));
    (*mq1)[0][0] = 0.1;
    (*mq1)[1][0] = 0.1;

    Matrix *mq2 = allocTmpMatrix(Shape(2, 1));
    (*mq2)[0][0] = 0.2;
    (*mq2)[1][0] = 0.2;

    autograd::Node *q1 = autograd::allocNode(mq1);
    autograd::Node *q2 = autograd::allocNode(mq2);

    q1->require_grad();
    q2->require_grad();

    queries.push_back(q1);
    queries.push_back(q2);

    std::vector<autograd::Node *> keys;

    Matrix *mk1 = allocTmpMatrix(Shape(2, 5));
    (*mk1)[0][0] = 1.1;
    (*mk1)[1][0] = 1.1;
    (*mk1)[0][1] = 1.2;
    (*mk1)[1][1] = 1.2;
    (*mk1)[0][2] = 1.3;
    (*mk1)[1][2] = 1.3;
    (*mk1)[0][3] = 1.4;
    (*mk1)[1][3] = 1.4;
    (*mk1)[0][4] = 1.5;
    (*mk1)[1][4] = 1.5;

    Matrix *mk2 = allocTmpMatrix(Shape(2, 5));
    (*mk2)[0][0] = 2.1;
    (*mk2)[1][0] = 2.1;
    (*mk2)[0][1] = 2.2;
    (*mk2)[1][1] = 2.2;
    (*mk2)[0][2] = 2.3;
    (*mk2)[1][2] = 2.3;
    (*mk2)[0][3] = 2.4;
    (*mk2)[1][3] = 2.4;
    (*mk2)[0][4] = 2.5;
    (*mk2)[1][4] = 2.5;

    autograd::Node *k1 = autograd::allocNode(mk1);
    autograd::Node *k2 = autograd::allocNode(mk2);

    k1->require_grad();
    k2->require_grad();

    keys.push_back(k1);
    keys.push_back(k2);

    std::vector<autograd::Node *> values;

    Matrix *mv1 = allocTmpMatrix(Shape(4, 5));
    (*mv1)[0][0] = 3.1;
    (*mv1)[1][0] = 3.1;
    (*mv1)[2][0] = 3.1;
    (*mv1)[3][0] = 3.1;
    (*mv1)[0][1] = 3.2;
    (*mv1)[1][1] = 3.2;
    (*mv1)[2][1] = 3.2;
    (*mv1)[3][1] = 3.2;
    (*mv1)[0][2] = 3.3;
    (*mv1)[1][2] = 3.3;
    (*mv1)[2][2] = 3.3;
    (*mv1)[3][2] = 3.3;
    (*mv1)[0][3] = 3.4;
    (*mv1)[1][3] = 3.4;
    (*mv1)[2][3] = 3.4;
    (*mv1)[3][3] = 3.4;
    (*mv1)[0][4] = 3.5;
    (*mv1)[1][4] = 3.5;
    (*mv1)[2][4] = 3.5;
    (*mv1)[3][4] = 3.5;

    Matrix *mv2 = allocTmpMatrix(Shape(4, 5));
    (*mv2)[0][0] = 4.1;
    (*mv2)[1][0] = 4.1;
    (*mv2)[2][0] = 4.1;
    (*mv2)[3][0] = 4.1;
    (*mv2)[0][1] = 4.2;
    (*mv2)[1][1] = 4.2;
    (*mv2)[2][1] = 4.2;
    (*mv2)[3][1] = 4.2;
    (*mv2)[0][2] = 4.3;
    (*mv2)[1][2] = 4.3;
    (*mv2)[2][2] = 4.3;
    (*mv2)[3][2] = 4.3;
    (*mv2)[0][3] = 4.4;
    (*mv2)[1][3] = 4.4;
    (*mv2)[2][3] = 4.4;
    (*mv2)[3][3] = 4.4;
    (*mv2)[0][4] = 4.5;
    (*mv2)[1][4] = 4.5;
    (*mv2)[2][4] = 4.5;
    (*mv2)[3][4] = 4.5;

    autograd::Node *v1 = autograd::allocNode(mv1);
    autograd::Node *v2 = autograd::allocNode(mv2);

    v1->require_grad();
    v2->require_grad();

    values.push_back(v1);
    values.push_back(v2);

    DotProductAttetion attention(0);

    std::vector<autograd::Node *> res = attention.forward(queries, keys, values, valid_lens);

    std::vector<uint> labels = {2, 3};

    autograd::Node *loss = autograd::cat(res, 0)->CrossEntropy(labels);

    cout << "res: " << endl;
    for (auto r : res) {
        cout << *r->get_weight() << endl;
    }

    cout << "loss: " << endl;
    cout << *loss->get_weight() << endl;

    loss->backward();

    cout << "q1 grad: " << endl;
    cout << *q1->get_grad() << endl;
    cout << "q2 grad: " << endl;
    cout << *q2->get_grad() << endl;

    cout << "k1 grad: " << endl;
    cout << *k1->get_grad() << endl;
    cout << "k2 grad: " << endl;
    cout << *k2->get_grad() << endl;

    cout << "v1 grad: " << endl;
    cout << *v1->get_grad() << endl;
    cout << "v2 grad: " << endl;
    cout << *v2->get_grad() << endl;

    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
}

void test_attention_without_mask() {

    std::vector<uint> valid_lens = {5, 5}; // all valid

    test_attention(valid_lens);
}

void test_attention_with_mask() {

    std::vector<uint> valid_lens = {2, 4}; // 3 valid for the first query

    test_attention(valid_lens);
}

void init_qkv_labels(
    std::vector<autograd::Node *> &queries,
    std::vector<autograd::Node *> &keys,
    std::vector<autograd::Node *> &values,
    std::vector<uint> &labels
) {
    Matrix *mq1 = allocTmpMatrix(Shape(2, 1));
    (*mq1)[0][0] = 0.1;
    (*mq1)[1][0] = 0.1;

    Matrix *mq2 = allocTmpMatrix(Shape(2, 1));
    (*mq2)[0][0] = 0.2;
    (*mq2)[1][0] = 0.2;

    autograd::Node *q1 = autograd::allocNode(mq1);
    autograd::Node *q2 = autograd::allocNode(mq2);

    q1->require_grad();
    q2->require_grad();

    queries.push_back(q1);
    queries.push_back(q2);

    Matrix *mk1 = allocTmpMatrix(Shape(2, 2));
    (*mk1)[0][0] = 1.1;
    (*mk1)[1][0] = 1.1;
    (*mk1)[0][1] = 1.2;
    (*mk1)[1][1] = 1.2;

    Matrix *mk2 = allocTmpMatrix(Shape(2, 2));
    (*mk2)[0][0] = 2.1;
    (*mk2)[1][0] = 2.1;
    (*mk2)[0][1] = 2.2;
    (*mk2)[1][1] = 2.2;

    
    autograd::Node *k1 = autograd::allocNode(mk1);
    autograd::Node *k2 = autograd::allocNode(mk2);

    k1->require_grad();
    k2->require_grad();

    keys.push_back(k1);
    keys.push_back(k2);

    Matrix *mv1 = allocTmpMatrix(Shape(2, 2));
    (*mv1)[0][0] = 3.1;
    (*mv1)[1][0] = 3.1;
    (*mv1)[0][1] = 3.2;
    (*mv1)[1][1] = 3.2;

    Matrix *mv2 = allocTmpMatrix(Shape(2, 2));
    (*mv2)[0][0] = 4.1;
    (*mv2)[1][0] = 4.1;
    (*mv2)[0][1] = 4.2;
    (*mv2)[1][1] = 4.2;

    autograd::Node *v1 = autograd::allocNode(mv1);
    autograd::Node *v2 = autograd::allocNode(mv2);

    v1->require_grad();
    v2->require_grad();

    values.push_back(v1);
    values.push_back(v2);

    labels.push_back(0);
    labels.push_back(1);
}

void print_qkv_res_grad(
    const std::vector<autograd::Node *> &queries,
    const std::vector<autograd::Node *> &keys,
    const std::vector<autograd::Node *> &values,
    const std::vector<autograd::Node *> &res
) {
    cout << "res : " << endl;
    for (auto r : res) {
        cout << *r->get_weight() << endl;
    }

    cout << "q1 grad: " << endl;
    cout << *queries[0]->get_grad() << endl;
    cout << "q2 grad: " << endl;
    cout << *queries[1]->get_grad() << endl;

    cout << "k1 grad: " << endl;
    cout << *keys[0]->get_grad() << endl;
    cout << "k2 grad: " << endl;
    cout << *keys[1]->get_grad() << endl;

    cout << "v1 grad: " << endl;
    cout << *values[0]->get_grad() << endl;
    cout << "v2 grad: " << endl;
    cout << *values[1]->get_grad() << endl;

    cout << "res grad: " << endl;
    for (auto r : res) {
        cout << *r->get_grad() << endl;
    }
}

void test_mh_attention(const std::vector<uint> &valid_lens) {
    std::vector<autograd::Node *> queries;
    std::vector<autograd::Node *> keys;
    std::vector<autograd::Node *> values;
    std::vector<uint> labels;
    init_qkv_labels(queries, keys, values, labels);
    MultiHeadAttention attention(1, 2, 0);
    std::vector<autograd::Node *> res = attention.forward(queries, keys, values, valid_lens);
    autograd::Node *loss = autograd::cat(res, 0)->CrossEntropy(labels);
    cout << "loss: " << endl;
    cout << *loss->get_weight() << endl;
    loss->backward();
    print_qkv_res_grad(queries, keys, values, res);
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
}

void test_attention1(const std::vector<uint> &valid_lens) {
    std::vector<autograd::Node *> queries;
    std::vector<autograd::Node *> keys;
    std::vector<autograd::Node *> values;
    std::vector<uint> labels;
    init_qkv_labels(queries, keys, values, labels);
    DotProductAttetion attention(0);
    std::vector<autograd::Node *> res = attention.forward(queries, keys, values, valid_lens);
    autograd::Node *loss = autograd::cat(res, 0)->CrossEntropy(labels);
    cout << "loss: " << endl;
    cout << *loss->get_weight() << endl;
    loss->backward();
    print_qkv_res_grad(queries, keys, values, res);
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
}

void test_attention_to_cp_with_mha() {
    std::vector<uint> valid_lens = {5, 5}; // all valid
    test_attention1(valid_lens);
}


void test_mh_attention_without_mask() {
    std::vector<uint> valid_lens = {5, 5}; // all valid
    test_mh_attention(valid_lens);
}

void test_lazy_liner() {

    std::vector<autograd::Node *> queries;

    Matrix *mq1 = allocTmpMatrix(Shape(2, 1));
    (*mq1)[0][0] = 0.1;
    (*mq1)[1][0] = 0.1;

    autograd::Node *q1 = autograd::allocNode(mq1);

    q1->require_grad();

    auto W = new autograd::LazyLiner(3, false);

    std::vector<autograd::Node *> res = {W->forward(q1)};

    std::vector<uint> labels = {2};

    autograd::Node *loss = autograd::cat(res, 0)->CrossEntropy(labels);

    cout << "res: " << endl;
    for (auto r : res) {
        cout << *r->get_weight() << endl;
    }

    cout << "loss: " << endl;
    cout << *loss->get_weight() << endl;

    loss->backward();

    cout << "q1 grad: " << endl;
    cout << *q1->get_grad() << endl;

    delete W;
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();  
}

int main() {
    // test_layernorm();
    // test_softmax();
    // test_attention_without_mask();
    // test_attention_with_mask();
    cout << "------ test_mh_attention_without_mask ------" << endl;
    test_mh_attention_without_mask();
    cout << "------ test_mh_attention_without_mask end ------" << endl;
    cout << "------ test_attention_to_cp_with_mha ------" << endl;
    test_attention_to_cp_with_mha();
    cout << "------ test_attention_to_cp_with_mha end ------" << endl;
    // test_lazy_liner();
    return 0;
}