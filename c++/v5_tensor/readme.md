## plan res

```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/v5_tensor$ ./mnist_cuda 
Tensors:
        Tensor(input)(3, 2)
        Tensor(w)(2, 2)
        Tensor(bias)(2)
        Tensor(INT8)(labels)(3)
        Tensor(res_at)(3, 2)
        Tensor(expand_add)(3, 2)
        Tensor(relu_res)(3, 2)
        Tensor(maxs)(3)
        Tensor(sums)(3)
        Tensor(cross_entropy)(1)
        Tensor(relu_prime)(3, 2)
        Tensor(grad_mul_relu_prime)(3, 2)
Tensor Views:
        Tensor(w_transpose_view)(2, 2)
        Tensor(input_transpose_view)(2, 3)
Grad Tensors:
        Tensor(input_grad)(3, 2)
        Tensor(w_grad)(2, 2)
        Tensor(bias_grad)(2)
        Tensor(res_at_grad)(3, 2)
        Tensor(expand_add_grad)(3, 2)
        Tensor(relu_res_grad)(3, 2)
        Tensor(cross_entropy_grad)(1)
        Tensor(sum_tmp)(2)
Actions:
AtAction: Tensor(input)(3, 2) at Tensor(w)(2, 2) -> Tensor(res_at)(3, 2)
ExpandAddAction: Tensor(res_at)(3, 2) + Tensor(bias)(2) -> Tensor(expand_add)(3, 2)
ReluAction: Tensor(expand_add)(3, 2) -> Tensor(relu_res)(3, 2)
CrossEntropyAction: Tensor(relu_res)(3, 2) with labels Tensor(INT8)(labels)(3) -> Tensor(cross_entropy)(1) context : Tensor(maxs)(3), Tensor(sums)(3)
CrossEntropyBackwardAction: Tensor(relu_res)(3, 2) with labels Tensor(INT8)(labels)(3) -> Tensor(relu_res_grad)(3, 2) context : Tensor(maxs)(3), Tensor(sums)(3)
ReluPrimeAction: Tensor(expand_add)(3, 2) -> Tensor(relu_prime)(3, 2)
MulAction: Tensor(relu_prime)(3, 2) * Tensor(relu_res_grad)(3, 2) -> Tensor(grad_mul_relu_prime)(3, 2)
AddEqAction: Tensor(expand_add_grad)(3, 2) += Tensor(grad_mul_relu_prime)(3, 2)
AddEqAction: Tensor(res_at_grad)(3, 2) += Tensor(expand_add_grad)(3, 2)
SumAction: Tensor(expand_add_grad)(3, 2) -> Tensor(sum_tmp)(2) along dim 0
AddEqAction: Tensor(bias_grad)(2) += Tensor(sum_tmp)(2)
AtAction: Tensor(res_at_grad)(3, 2) at Tensor(w_transpose_view)(2, 2) -> Tensor(input_grad)(3, 2)
AtAction: Tensor(input_transpose_view)(2, 3) at Tensor(res_at_grad)(3, 2) -> Tensor(w_grad)(2, 2)
```

## cpu 单核表现

```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/v5_tensor$ time ./mnist_cuda -g 0
images magic : 2051
label magic : 2049
lables_num : 60000
data loaded.
Actions:
[once]InitWeightAction: initializing Tensor(w1)(784, 30) with type gauss
[once]InitWeightAction: initializing Tensor(w2)(30, 10) with type gauss
AtAction: Tensor(inputs)(100, 784) at Tensor(w1)(784, 30) -> Tensor(res_at)(100, 30)
ExpandAddAction: Tensor(res_at)(100, 30) + Tensor(bias1)(30) -> Tensor(expand_add)(100, 30)
ReluAction: Tensor(expand_add)(100, 30) -> Tensor(relu_res)(100, 30)
AtAction: Tensor(relu_res)(100, 30) at Tensor(w2)(30, 10) -> Tensor(res_at)(100, 10)
ExpandAddAction: Tensor(res_at)(100, 10) + Tensor(bias2)(10) -> Tensor(expand_add)(100, 10)
CrossEntropyAction: Tensor(expand_add)(100, 10) with labels Tensor(INT32)(labels)(100) -> Tensor(cross_entropy)(1) context : Tensor(maxs)(100), Tensor(sums)(100)
ZeroGradAction: zeroing gradients
============= BoundaryAction: boundary action =============
CrossEntropyBackwardAction: Tensor(expand_add)(100, 10) with labels Tensor(INT32)(labels)(100) -> Tensor(expand_add_grad)(100, 10) context : Tensor(maxs)(100), Tensor(sums)(100)
AddEqAction: Tensor(res_at_grad)(100, 10) += Tensor(expand_add_grad)(100, 10)
SumAction: Tensor(expand_add_grad)(100, 10) -> Tensor(sum_tmp)(10) along dim 0
AddEqAction: Tensor(bias2_grad)(10) += Tensor(sum_tmp)(10)
AtAction: Tensor(res_at_grad)(100, 10) at Tensor(w2_transpose_view)(10, 30) -> Tensor(relu_res_grad)(100, 30)
AtAction: Tensor(relu_res_transpose_view)(30, 100) at Tensor(res_at_grad)(100, 10) -> Tensor(w2_grad)(30, 10)
ReluPrimeAction: Tensor(expand_add)(100, 30) -> Tensor(relu_prime)(100, 30)
MulAction: Tensor(relu_prime)(100, 30) * Tensor(relu_res_grad)(100, 30) -> Tensor(grad_mul_relu_prime)(100, 30)
AddEqAction: Tensor(expand_add_grad)(100, 30) += Tensor(grad_mul_relu_prime)(100, 30)
AddEqAction: Tensor(res_at_grad)(100, 30) += Tensor(expand_add_grad)(100, 30)
SumAction: Tensor(expand_add_grad)(100, 30) -> Tensor(sum_tmp)(30) along dim 0
AddEqAction: Tensor(bias1_grad)(30) += Tensor(sum_tmp)(30)
AtAction: Tensor(inputs_transpose_view)(784, 100) at Tensor(res_at_grad)(100, 30) -> Tensor(w1_grad)(784, 30)
CalcAllGradNormAction: calculating gradient norm for 4 tensors -> Tensor(clip_grad_norm)(1)
ClipGradAction: clipping gradient Tensor(w1_grad)(784, 30) with norm Tensor(clip_grad_norm)(1) to grad_clip_val: 1
ClipGradAction: clipping gradient Tensor(w2_grad)(30, 10) with norm Tensor(clip_grad_norm)(1) to grad_clip_val: 1
ClipGradAction: clipping gradient Tensor(bias1_grad)(30) with norm Tensor(clip_grad_norm)(1) to grad_clip_val: 1
ClipGradAction: clipping gradient Tensor(bias2_grad)(10) with norm Tensor(clip_grad_norm)(1) to grad_clip_val: 1
AdamStepAction: updating parameter Tensor(w1)(784, 30) with learning rate 0.001
AdamStepAction: updating parameter Tensor(w2)(30, 10) with learning rate 0.001
AdamStepAction: updating parameter Tensor(bias1)(30) with learning rate 0.001
AdamStepAction: updating parameter Tensor(bias2)(10) with learning rate 0.001
epoch : 0 [50000/50000] loss : 0.386509
evaluating :  [10000/10000] correct : 9303
epoch : 1 [50000/50000] loss : 0.230633
evaluating :  [10000/10000] correct : 9447
epoch : 2 [50000/50000] loss : 0.197479
evaluating :  [10000/10000] correct : 9465
epoch : 3 [50000/50000] loss : 0.184908
evaluating :  [10000/10000] correct : 9487
epoch : 4 [50000/50000] loss : 0.171148
evaluating :  [10000/10000] correct : 9501
epoch : 5 [50000/50000] loss : 0.162561
evaluating :  [10000/10000] correct : 9531
epoch : 6 [50000/50000] loss : 0.156568
evaluating :  [10000/10000] correct : 9484
epoch : 7 [50000/50000] loss : 0.152769
evaluating :  [10000/10000] correct : 9525
epoch : 8 [50000/50000] loss : 0.151582
evaluating :  [10000/10000] correct : 9478
epoch : 9 [50000/50000] loss : 0.142558
evaluating :  [10000/10000] correct : 9498

real    0m36.432s
user    0m36.344s
sys     0m0.078s
```

## gpu 表现
```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/v5_tensor$ time ./mnist_cuda 
images magic : 2051
label magic : 2049
lables_num : 60000
data loaded.
Actions:
[once]InitWeightAction: initializing Tensor(w1)(784, 30) with type gauss
[once]InitWeightAction: initializing Tensor(w2)(30, 10) with type gauss
AtAction: Tensor(inputs)(100, 784) at Tensor(w1)(784, 30) -> Tensor(res_at)(100, 30)
ExpandAddAction: Tensor(res_at)(100, 30) + Tensor(bias1)(30) -> Tensor(expand_add)(100, 30)
ReluAction: Tensor(expand_add)(100, 30) -> Tensor(relu_res)(100, 30)
AtAction: Tensor(relu_res)(100, 30) at Tensor(w2)(30, 10) -> Tensor(res_at)(100, 10)
ExpandAddAction: Tensor(res_at)(100, 10) + Tensor(bias2)(10) -> Tensor(expand_add)(100, 10)
CrossEntropyAction: Tensor(expand_add)(100, 10) with labels Tensor(INT32)(labels)(100) -> Tensor(cross_entropy)(1) context : Tensor(maxs)(100), Tensor(sums)(100)
ZeroGradAction: zeroing gradients
============= BoundaryAction: boundary action =============
CrossEntropyBackwardAction: Tensor(expand_add)(100, 10) with labels Tensor(INT32)(labels)(100) -> Tensor(expand_add_grad)(100, 10) context : Tensor(maxs)(100), Tensor(sums)(100)
AddEqAction: Tensor(res_at_grad)(100, 10) += Tensor(expand_add_grad)(100, 10)
SumAction: Tensor(expand_add_grad)(100, 10) -> Tensor(sum_tmp)(10) along dim 0
AddEqAction: Tensor(bias2_grad)(10) += Tensor(sum_tmp)(10)
AtAction: Tensor(res_at_grad)(100, 10) at Tensor(w2_transpose_view)(10, 30) -> Tensor(relu_res_grad)(100, 30)
AtAction: Tensor(relu_res_transpose_view)(30, 100) at Tensor(res_at_grad)(100, 10) -> Tensor(w2_grad)(30, 10)
ReluPrimeAction: Tensor(expand_add)(100, 30) -> Tensor(relu_prime)(100, 30)
MulAction: Tensor(relu_prime)(100, 30) * Tensor(relu_res_grad)(100, 30) -> Tensor(grad_mul_relu_prime)(100, 30)
AddEqAction: Tensor(expand_add_grad)(100, 30) += Tensor(grad_mul_relu_prime)(100, 30)
AddEqAction: Tensor(res_at_grad)(100, 30) += Tensor(expand_add_grad)(100, 30)
SumAction: Tensor(expand_add_grad)(100, 30) -> Tensor(sum_tmp)(30) along dim 0
AddEqAction: Tensor(bias1_grad)(30) += Tensor(sum_tmp)(30)
AtAction: Tensor(inputs_transpose_view)(784, 100) at Tensor(res_at_grad)(100, 30) -> Tensor(w1_grad)(784, 30)
CalcAllGradNormAction: calculating gradient norm for 4 tensors -> Tensor(clip_grad_norm)(1)
ClipGradAction: clipping gradient Tensor(w1_grad)(784, 30) with norm Tensor(clip_grad_norm)(1) to grad_clip_val: 1
ClipGradAction: clipping gradient Tensor(w2_grad)(30, 10) with norm Tensor(clip_grad_norm)(1) to grad_clip_val: 1
ClipGradAction: clipping gradient Tensor(bias1_grad)(30) with norm Tensor(clip_grad_norm)(1) to grad_clip_val: 1
ClipGradAction: clipping gradient Tensor(bias2_grad)(10) with norm Tensor(clip_grad_norm)(1) to grad_clip_val: 1
AdamStepAction: updating parameter Tensor(w1)(784, 30) with learning rate 0.001
AdamStepAction: updating parameter Tensor(w2)(30, 10) with learning rate 0.001
AdamStepAction: updating parameter Tensor(bias1)(30) with learning rate 0.001
AdamStepAction: updating parameter Tensor(bias2)(10) with learning rate 0.001
epoch : 0 [50000/50000] loss : 0.380846
evaluating :  [10000/10000] correct : 9414
epoch : 1 [50000/50000] loss : 0.216472
evaluating :  [10000/10000] correct : 9422
epoch : 2 [50000/50000] loss : 0.183812
evaluating :  [10000/10000] correct : 9524
epoch : 3 [50000/50000] loss : 0.176573
evaluating :  [10000/10000] correct : 9487
epoch : 4 [50000/50000] loss : 0.163809
evaluating :  [10000/10000] correct : 9497
epoch : 5 [50000/50000] loss : 0.155687
evaluating :  [10000/10000] correct : 9490
epoch : 6 [50000/50000] loss : 0.146089
evaluating :  [10000/10000] correct : 9493
epoch : 7 [50000/50000] loss : 0.139883
evaluating :  [10000/10000] correct : 9503
epoch : 8 [50000/50000] loss : 0.14207
evaluating :  [10000/10000] correct : 9516
epoch : 9 [50000/50000] loss : 0.134731
evaluating :  [10000/10000] correct : 9511

real    0m2.328s
user    0m1.940s
sys     0m0.358s
```

## 20250528 transformer loss可以收敛了

```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/v5_tensor$ ./transformer 
enc_vocab_size : 7939
dec_vocab_size : 13387
bos_id : 3
eos_id : 1
src_pad_id : 0
tgt_pad_id : 0
epoch 0 :  [167040/167130]loss : 4.69666
epoch 1 :  [167040/167130]loss : 3.93095
```

# checkpoint test

```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/v5_tensor$ ./transformer -f ../../resources/fra_tiny.txt 
corpus : ../../resources/fra_tiny.txt
epochs : 10
batch_size : 128
gpu : 1
learning rate : 0.001
checkpoint : 
enc_vocab_size : 7939
dec_vocab_size : 13387
bos_id : 3
eos_id : 1
src_pad_id : 0
tgt_pad_id : 0
epoch 0 :  [384/384]loss : 9.27215
epoch 1 :  [384/384]loss : 8.31903
epoch 2 :  [384/384]loss : 7.43319
epoch 3 :  [384/384]loss : 6.61569
epoch 4 :  [384/384]loss : 5.94754
epoch 5 :  [384/384]loss : 5.45714
epoch 6 :  [384/384]loss : 5.1212
epoch 7 :  [384/384]loss : 4.92025
epoch 8 :  [384/384]loss : 4.82687
epoch 9 :  [384/384]loss : 4.78579
checkpoint saved : ./checkpoints/checkpoint_20250528_215330_9.bin

(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/v5_tensor$ ./transformer -f ../../resources/fra_tiny.txt -c ./checkpoints/checkpoint_20250528_215330_9.bin
corpus : ../../resources/fra_tiny.txt
epochs : 10
batch_size : 128
gpu : 1
learning rate : 0.001
checkpoint : ./checkpoints/checkpoint_20250528_215330_9.bin
enc_vocab_size : 7939
dec_vocab_size : 13387
bos_id : 3
eos_id : 1
src_pad_id : 0
tgt_pad_id : 0
loading from checkpoint : ./checkpoints/checkpoint_20250528_215330_9.bin
loaded from checkpoint
epoch 0 :  [384/384]loss : 4.76779
epoch 1 :  [384/384]loss : 4.73401
epoch 2 :  [384/384]loss : 4.68025
epoch 3 :  [384/384]loss : 4.59945
epoch 4 :  [384/384]loss : 4.50977
epoch 5 :  [384/384]loss : 4.46144
epoch 6 :  [384/384]loss : 4.34954
epoch 7 :  [384/384]loss : 4.27593
epoch 8 :  [384/384]loss : 4.19153
epoch 9 :  [384/384]loss : 4.11973
checkpoint saved : ./checkpoints/checkpoint_20250528_215401_9.bin
```

# 修正 dropout bp bug后 tiny数据集上loss下降正常

```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/v5_tensor$ ./transformer -f ../../resources/fra_tiny.txt -e 100
corpus : ../../resources/fra_tiny.txt
epochs : 100
batch_size : 128
gpu : 1
learning rate : 0.001
checkpoint : 
enc_vocab_size : 7939
dec_vocab_size : 13387
bos_id : 3
eos_id : 1
src_pad_id : 0
tgt_pad_id : 0
epoch 0 :  [384/384]loss : 9.22685
epoch 1 :  [384/384]loss : 8.17875
epoch 2 :  [384/384]loss : 7.31543
epoch 3 :  [384/384]loss : 6.55572
epoch 4 :  [384/384]loss : 5.94124
epoch 5 :  [384/384]loss : 5.46998
epoch 6 :  [384/384]loss : 5.1335
epoch 7 :  [384/384]loss : 4.91032
epoch 8 :  [384/384]loss : 4.74691
epoch 9 :  [384/384]loss : 4.65693
epoch 10 :  [384/384]loss : 4.54825
epoch 11 :  [384/384]loss : 4.43991
epoch 12 :  [384/384]loss : 4.28683
epoch 13 :  [384/384]loss : 4.13971
epoch 14 :  [384/384]loss : 4.01758
epoch 15 :  [384/384]loss : 3.91402
epoch 16 :  [384/384]loss : 3.76348
epoch 17 :  [384/384]loss : 3.68659
epoch 18 :  [384/384]loss : 3.49679
epoch 19 :  [384/384]loss : 3.37338
epoch 20 :  [384/384]loss : 3.29745
epoch 21 :  [384/384]loss : 3.15076
epoch 22 :  [384/384]loss : 3.02548
epoch 23 :  [384/384]loss : 2.90251
epoch 24 :  [384/384]loss : 2.77181
epoch 25 :  [384/384]loss : 2.68067
epoch 26 :  [384/384]loss : 2.5364
epoch 27 :  [384/384]loss : 2.43822
epoch 28 :  [384/384]loss : 2.31102
epoch 29 :  [384/384]loss : 2.19813
epoch 30 :  [384/384]loss : 2.08746
epoch 31 :  [384/384]loss : 1.99021
epoch 32 :  [384/384]loss : 1.88037
epoch 33 :  [384/384]loss : 1.80944
epoch 34 :  [384/384]loss : 1.68084
epoch 35 :  [384/384]loss : 1.60938
epoch 36 :  [384/384]loss : 1.51168
epoch 37 :  [384/384]loss : 1.43711
epoch 38 :  [384/384]loss : 1.35525
epoch 39 :  [384/384]loss : 1.29163
epoch 40 :  [384/384]loss : 1.21349
epoch 41 :  [384/384]loss : 1.15099
epoch 42 :  [384/384]loss : 1.07385
epoch 43 :  [384/384]loss : 1.03574
epoch 44 :  [384/384]loss : 0.976446
epoch 45 :  [384/384]loss : 0.922481
epoch 46 :  [384/384]loss : 0.871867
epoch 47 :  [384/384]loss : 0.843808
epoch 48 :  [384/384]loss : 0.787574
epoch 49 :  [384/384]loss : 0.737342
epoch 50 :  [384/384]loss : 0.692409
epoch 51 :  [384/384]loss : 0.658162
epoch 52 :  [384/384]loss : 0.625736
epoch 53 :  [384/384]loss : 0.600884
epoch 54 :  [384/384]loss : 0.570778
epoch 55 :  [384/384]loss : 0.545031
epoch 56 :  [384/384]loss : 0.517207
epoch 57 :  [384/384]loss : 0.49355
epoch 58 :  [384/384]loss : 0.472806
epoch 59 :  [384/384]loss : 0.460372
epoch 60 :  [384/384]loss : 0.427689
epoch 61 :  [384/384]loss : 0.413875
epoch 62 :  [384/384]loss : 0.396615
epoch 63 :  [384/384]loss : 0.372431
epoch 64 :  [384/384]loss : 0.368165
epoch 65 :  [384/384]loss : 0.353622
epoch 66 :  [384/384]loss : 0.342993
epoch 67 :  [384/384]loss : 0.326296
epoch 68 :  [384/384]loss : 0.321928
epoch 69 :  [384/384]loss : 0.307301
epoch 70 :  [384/384]loss : 0.310316
epoch 71 :  [384/384]loss : 0.303165
epoch 72 :  [384/384]loss : 0.286861
epoch 73 :  [384/384]loss : 0.2856
epoch 74 :  [384/384]loss : 0.277225
epoch 75 :  [384/384]loss : 0.277469
epoch 76 :  [384/384]loss : 0.265212
epoch 77 :  [384/384]loss : 0.253928
epoch 78 :  [384/384]loss : 0.265037
epoch 79 :  [384/384]loss : 0.256729
epoch 80 :  [384/384]loss : 0.252929
epoch 81 :  [384/384]loss : 0.248567
epoch 82 :  [384/384]loss : 0.247802
epoch 83 :  [384/384]loss : 0.23653
epoch 84 :  [384/384]loss : 0.234957
epoch 85 :  [384/384]loss : 0.23887
epoch 86 :  [384/384]loss : 0.238759
epoch 87 :  [384/384]loss : 0.235842
epoch 88 :  [384/384]loss : 0.235275
epoch 89 :  [384/384]loss : 0.233121
epoch 90 :  [384/384]loss : 0.223154
epoch 91 :  [384/384]loss : 0.221809
epoch 92 :  [384/384]loss : 0.225124
epoch 93 :  [384/384]loss : 0.22043
epoch 94 :  [384/384]loss : 0.215261
epoch 95 :  [384/384]loss : 0.219092
epoch 96 :  [384/384]loss : 0.216041
epoch 97 :  [384/384]loss : 0.210444
epoch 98 :  [384/384]loss : 0.221074
epoch 99 :  [384/384]loss : 0.217064
checkpoint saved : ./checkpoints/checkpoint_20250529_144051_99.bin
```

# 修正dropout bp bug后重新测试单epoch耗时 step=32 batchsize=128
```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/v5_tensor$ time ./transformer -e 1
corpus : ../../resources/fra_preprocessed.txt
epochs : 1
batch_size : 128
gpu : 1
learning rate : 0.001
checkpoint : 
enc_vocab_size : 7939
dec_vocab_size : 13387
bos_id : 3
eos_id : 1
src_pad_id : 0
tgt_pad_id : 0
epoch 0 :  [167040/167130]loss : 4.05721
checkpoint saved : ./checkpoints/checkpoint_20250529_143558_0.bin

real    12m56.637s
user    12m55.901s
sys     0m0.734s
```

# prediction
```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/v5_tensor$ ./transformer -e 0 -c ./checkpoints/checkpoint_20250529_144051_99.bin
corpus : ../../resources/fra_preprocessed.txt
epochs : 0
batch_size : 128
gpu : 1
learning rate : 0.001
checkpoint : ./checkpoints/checkpoint_20250529_144051_99.bin
enc_vocab_size : 7939
dec_vocab_size : 13387
bos_id : 3
eos_id : 1
src_pad_id : 0
tgt_pad_id : 0
predicting : true
batch_size : 1
loading from checkpoint : ./checkpoints/checkpoint_20250529_144051_99.bin
loaded from checkpoint
serving mode
test file : ./test.txt
go now . -> allez-y maintenant . 
i know that it is highly unlikely that you'd ever want to go out -> je sais qu'il est hautement improbable que tu veuilles jamais sortir avec moi , mais j'ai tout de même besoin de demander au moins une fois . 
good job -> bon boulot ! 
how nice ! -> comme c'est du joli ! 
```

### mnist_cuda 加上 norm
```
epoch : 0 [50000/50000] loss : 0.67828
evaluating :  [10000/10000] correct : 9298
epoch : 1 [50000/50000] loss : 0.263307
evaluating :  [10000/10000] correct : 9466
epoch : 2 [50000/50000] loss : 0.195321
evaluating :  [10000/10000] correct : 9531
epoch : 3 [50000/50000] loss : 0.159109
evaluating :  [10000/10000] correct : 9593
epoch : 4 [50000/50000] loss : 0.135835
evaluating :  [10000/10000] correct : 9628
epoch : 5 [50000/50000] loss : 0.118771
evaluating :  [10000/10000] correct : 9650
epoch : 6 [50000/50000] loss : 0.105423
evaluating :  [10000/10000] correct : 9654
epoch : 7 [50000/50000] loss : 0.0952094
evaluating :  [10000/10000] correct : 9657
epoch : 8 [50000/50000] loss : 0.0868837
evaluating :  [10000/10000] correct : 9651
epoch : 9 [50000/50000] loss : 0.0799941
evaluating :  [10000/10000] correct : 9651
```

### mnist_cuda 不加 norm
```
epoch : 0 [50000/50000] loss : 0.390219
evaluating :  [10000/10000] correct : 9337
epoch : 1 [50000/50000] loss : 0.223057
evaluating :  [10000/10000] correct : 9432
epoch : 2 [50000/50000] loss : 0.199496
evaluating :  [10000/10000] correct : 9453
epoch : 3 [50000/50000] loss : 0.178141
evaluating :  [10000/10000] correct : 9445
epoch : 4 [50000/50000] loss : 0.165149
evaluating :  [10000/10000] correct : 9451
epoch : 5 [50000/50000] loss : 0.156534
evaluating :  [10000/10000] correct : 9476
epoch : 6 [50000/50000] loss : 0.151771
evaluating :  [10000/10000] correct : 9436
epoch : 7 [50000/50000] loss : 0.145766
evaluating :  [10000/10000] correct : 9442
epoch : 8 [50000/50000] loss : 0.140849
evaluating :  [10000/10000] correct : 9475
epoch : 9 [50000/50000] loss : 0.137518
evaluating :  [10000/10000] correct : 9464
```