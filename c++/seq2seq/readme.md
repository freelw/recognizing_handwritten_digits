# seq2seq

### training

```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/seq2seq$ time ./seq2seq -e 50 -c ./checkpoints/checkpoint_20250321_174358_9.bin
OMP_THREADS: 8
epochs : 50
dropout : 0.2
lr : 0.005
tiny : 0
data loaded
loading from checkpoint : ./checkpoints/checkpoint_20250321_174358_9.bin
loaded from checkpoint
[167130/167130]checkpoint saved : ./checkpoints/checkpoint_20250322_085920_0.bin
epoch 0 loss : 1.21077 emit_clip : 0
[167130/167130]epoch 1 loss : 1.21596 emit_clip : 3
[167130/167130]epoch 2 loss : 1.20476 emit_clip : 2
[167130/167130]epoch 3 loss : 1.19928 emit_clip : 3
[167130/167130]epoch 4 loss : 1.19412 emit_clip : 3
[167130/167130]epoch 5 loss : 1.1944 emit_clip : 3
[167130/167130]epoch 6 loss : 1.18376 emit_clip : 3
[167130/167130]epoch 7 loss : 1.1833 emit_clip : 3
[167130/167130]epoch 8 loss : 1.18186 emit_clip : 3
[167130/167130]epoch 9 loss : 1.17589 emit_clip : 3
[167130/167130]checkpoint saved : ./checkpoints/checkpoint_20250322_085920_10.bin
epoch 10 loss : 1.17234 emit_clip : 3
[167130/167130]epoch 11 loss : 1.17417 emit_clip : 4
[167130/167130]epoch 12 loss : 1.16851 emit_clip : 4
[167130/167130]epoch 13 loss : 1.17347 emit_clip : 2
[50048/167130]
```

### 推理
```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/seq2seq$ ./seq2seq -e 0 -c ./checkpoints/checkpoint_20250322_085920_10.bin
OMP_THREADS: 8
epochs : 0
dropout : 0.2
lr : 0.005
tiny : 0
data loaded
loading from checkpoint : ./checkpoints/checkpoint_20250322_085920_10.bin
loaded from checkpoint
serving mode
go now . <eos>
translate res : allez-y maintenant , maintenant , avant que partiez ! <eos>
i try . <eos>
translate res : j'essaie de <unk> régulier , essayons de me dérober autour de ce que j'ai essayé de la facture . <eos>
cheers ! <eos>
translate res : félicitations ! <eos>
get up . <eos>
translate res : lève-toi là-haut ! <eos>
hug me . <eos>
translate res : serre-moi me <unk> dans ton bras de la chambre ! <eos>
i know . <eos>
translate res : je sais que je <unk> de la civilisation , je ne veux pas te faire . <eos>
no way ! <eos>
translate res : <unk> ! <eos>
be nice . <eos>
translate res : soyez gentils ! <eos>
i jumped . <eos>
translate res : j'ai sauté sauté par la dent <unk> . <eos>
congratulations ! <eos>
translate res : félicitations de manière félicitations ! <eos>
```

### bug 现象记录
1. cat1 没有传导grad
    这时loss依然能够有一定程度的下降，因为反向传播在decoder还工作，但是在cat encoder的ctx和tgt的embedding时断掉了
2. encoder forward token参数传递错误
    这时不能通过全部数据进行调整，所以loss下降到一定程度就不下降了
3. labels的顺序弄错了, 应该是每个step中的token紧挨着，而不是每个句子中的token紧挨着
4. 擅自把hidden和embedding减少到32的效果：模型不易收敛，看起来是表达能力不足，256收敛比较快
5. loss_sum 没有除以次数，fixed

### 关于loss

loss=1 是，判断正确的概率是 1/e 约为 36%

想要判断一个token的正确率在50%以上，需要loss下降到 -ln(0.5)以下，也就是0.69以下

想要正确率超过90%，loss需要小于0.105

### 关于参数初始化

由于引入了deep rnn，梯度的问题变得明显
需要用xavier方法初始化weight，针对tanh和sigmoid之前的层初始化的标准差有区别，见代码

### perf

计算瓶颈在矩阵乘法

![alt text](p_1473505.svg)

### checkpoint

./checkpoints/checkpoint_20250321_073021_5.bin