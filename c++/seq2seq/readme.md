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