# 基于动态图的 RNN 语言模型

### batch = 1024 速度提升很明显
```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/rnn_v1$ ./rnn -e 30 -c ./checkpoints/checkpoint_20250312_232157_3.bin
epochs : 30
train by ../../resources/timemachine_preprocessed.txt
epochs : 30
Data loaded
loading from checkpoint : ./checkpoints/checkpoint_20250312_232157_3.bin
loaded from checkpoint
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250312_233314_0.bin
epoch 0 loss : 2.2805709838867 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250312_233314_1.bin
epoch 1 loss : 2.2215175628662 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250312_233314_2.bin
epoch 2 loss : 2.1762795448303 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250312_233314_3.bin
epoch 3 loss : 2.1358518600464 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250312_233314_4.bin
epoch 4 loss : 2.0993001461029 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250312_233314_5.bin
epoch 5 loss : 2.0685639381409 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250312_233314_6.bin
epoch 6 loss : 2.0434064865112 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250312_233314_7.bin
epoch 7 loss : 2.0222582817078 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250312_233314_8.bin
epoch 8 loss : 2.0040545463562 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250312_233314_9.bin
epoch 9 loss : 1.9881755113602 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250312_233314_10.bin
epoch 10 loss : 1.9742200374603 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250312_233314_11.bin
epoch 11 loss : 1.9618582725525 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250312_233314_12.bin
epoch 12 loss : 1.9507826566696 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250312_233314_13.bin
epoch 13 loss : 1.9407542943954 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250312_233314_14.bin
epoch 14 loss : 1.931591629982 emit_clip : 0
[83968/173396]^Ccheckpoint saved : ./checkpoints/checkpoint_20250312_233314_15.bin
```