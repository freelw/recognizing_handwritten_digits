# 以自动构建计算图的方式实现手写识别

### forward代码片段
```
Node *MLP::forward(Node *input) {
    auto Z1 = W1->at(input)->expand_add(b1)->Relu();
    auto Z2 = W2->at(Z1)->expand_add(b2);
    return Z2;
}
```

### 训练代码片段
```
optimizer.zero_grad();
auto loss = m.forward(autograd::allocNode(input))->CrossEntropy(labels);
assert(loss->get_weight()->getShape().rowCnt == 1);
assert(loss->get_weight()->getShape().colCnt == 1);
DATATYPE ret = *(loss->get_weight())[0][0];
loss->backward();
optimizer.step();
```

### 执行效果

```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/v3$ time ./autogradtest 10 128 0 1
eval : 1
images magic : 2051
label magic : 2049
lables_num : 60000
data loaded.
epoch : [1/10] loss : 0.379095
9270 / 10000
epoch : [2/10] loss : 0.205186
9525 / 10000
epoch : [3/10] loss : 0.169757
9481 / 10000
epoch : [4/10] loss : 0.149764
9514 / 10000
epoch : [5/10] loss : 0.139201
9459 / 10000
epoch : [6/10] loss : 0.137201
9570 / 10000
epoch : [7/10] loss : 0.130763
9485 / 10000
epoch : [8/10] loss : 0.121712
9564 / 10000
epoch : [9/10] loss : 0.116632
9569 / 10000
epoch : [10/10] loss : 0.119332
9533 / 10000

real    0m26.839s
user    1m28.420s
sys     0m0.369s
```