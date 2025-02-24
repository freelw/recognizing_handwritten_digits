# 自动微分版本

```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/v1$ time ./recognizing_handwritten_digits_v1
images magic : 2051
label magic : 2049
lables_num : 60000
data loaded.
epoch : [1/30] update_mini_batch : [1000/5000]
epoch : [1/30] update_mini_batch : [2000/5000]
epoch : [1/30] update_mini_batch : [3000/5000]
epoch : [1/30] update_mini_batch : [4000/5000]
epoch : [1/30] update_mini_batch : [5000/5000]
correct: 8982 / 10000 loss: 0.5461937535
```

![alt text](1.svg)

第一个版本 vector 的 push_back 的消耗没有必要，直接用数组替换

![alt text](2.svg)

看起来好了不少