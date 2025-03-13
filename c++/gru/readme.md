# 基于动态图的 gru 语言模型


### forward 代码片段

backward自动计算，无需显示实现

```
for (auto input : inputs) {
    Node *r = (*(Wxr->at(input)) + Whr->at(hidden))->expand_add(Br)->Sigmoid();
    Node *z = (*(Wxz->at(input)) + Whz->at(hidden))->expand_add(Bz)->Sigmoid();
    Node *h_tilde = (*(Wxh->at(input)) + Whh->at(*r * hidden))->expand_add(Bh)->Tanh();
    hidden = *(*z * hidden) + *(1 - *z) * h_tilde;
    res.push_back(hidden);
}
```

### 执行8轮后推理效果

```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/gru$ ./gru -e 0 -c ./checkpoints/checkpoint_20250313_111049_7.bin
OMP_THREADS: 8
epochs : 0
train by ../../resources/timemachine_preprocessed.txt
epochs : 0
Data loaded
loading from checkpoint : ./checkpoints/checkpoint_20250313_111049_7.bin
loaded from checkpoint
serving mode
prefix : time traveller
predicted :  the the the the the the the t
prefix : the time machine
predicted :  the the the the the the the t
prefix : expounding a recondite
predicted :  the the the the the the the t
prefix :  traveller for so
predicted : r the the the the the the the
prefix : it has
predicted :  the the the the the the the t
prefix : so most people
predicted :  the the the the the the the t
prefix : is simply
predicted : the the the the the the the th
prefix :  we cannot move about
predicted :  and the the the the the the t
prefix : and the still
predicted :  the the the the the the the t

```