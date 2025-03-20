# seq2seq

### training
loss 可以收敛
```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/seq2seq$ ./seq2seq -e 5 -f ../../resources/fra_tiny.txt 
OMP_THREADS: 8
epochs : 5
data loaded
[300/300]checkpoint saved : ./checkpoints/checkpoint_20250319_172123_0.bin
epoch 0 loss : 27.9319 emit_clip : 1
[300/300]checkpoint saved : ./checkpoints/checkpoint_20250319_172123_1.bin
epoch 1 loss : 16.1145 emit_clip : 3
[300/300]checkpoint saved : ./checkpoints/checkpoint_20250319_172123_2.bin
epoch 2 loss : 11.9372 emit_clip : 3
[300/300]checkpoint saved : ./checkpoints/checkpoint_20250319_172123_3.bin
epoch 3 loss : 11.8653 emit_clip : 3
[300/300]checkpoint saved : ./checkpoints/checkpoint_20250319_172123_4.bin
epoch 4 loss : 11.6869 emit_clip : 3
```

### bug 现象记录
1. cat1 没有传导grad
    这时loss依然能够有一定程度的下降，因为反向传播在decoder还工作，但是在cat encoder的ctx和tgt的embedding时断掉了
2. encoder forward token参数传递错误
    这时不能通过全部数据进行调整，所以loss下降到一定程度就不下降了

### 关于loss

loss=1 是，判断正确的概率是 1/e 约为 36%

想要判断一个token的正确率在50%以上，需要loss下降到 -ln(0.5)以下，也就是0.69以下

想要正确率超过90%，loss需要小于0.105

### 关于参数初始化

由于引入了deep rnn，梯度的问题变得明显

需要用xavier方法初始化weight，针对tanh和sigmoid之前的层初始化的标准差有区别，见代码