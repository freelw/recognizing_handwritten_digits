# seq2seq

### training
loss 可以收敛
```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/seq2seq$ ./seq2seq -e 100 -f ../../resources/fra_stiny.txt
OMP_THREADS: 8
epochs : 100
dropout : 0.2
lr : 0.005
tiny : 0
data loaded
[2/2]checkpoint saved : ./checkpoints/checkpoint_20250320_191840_0.bin
epoch 0 loss : 9.5485 emit_clip : 1
...
[2/2]epoch 24 loss : 0.120336 emit_clip : 0
[2/2]epoch 25 loss : 0.0811348 emit_clip : 0
[2/2]epoch 26 loss : 0.0729082 emit_clip : 0
[2/2]epoch 27 loss : 0.0535464 emit_clip : 0
[2/2]epoch 28 loss : 0.0403591 emit_clip : 0
[2/2]checkpoint saved : ./checkpoints/checkpoint_20250320_191840_29.bin
```

### 推理
```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/seq2seq$ ./seq2seq -e 0 -c ./checkpoints/checkpoint_20250320_191840_29.bin
OMP_THREADS: 8
epochs : 0
dropout : 0.2
lr : 0.005
tiny : 0
data loaded
loading from checkpoint : ./checkpoints/checkpoint_20250320_191840_29.bin
loaded from checkpoint
serving mode
go now . <eos>
translate res : allez-y maintenant . <eos>
i try . <eos>
translate res : j'essaye . <eos>
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