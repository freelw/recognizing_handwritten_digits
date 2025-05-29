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

# bug

```
g meta : Tensor[31786](w_v_w_linear_grad)(256, 256)

从这里开始往回的导数都是0
```

node meta : Tensor[34894](div_res_reshape_sequence_mask_reshape)(512, 9, 9)
grad meta : Tensor[34895](div_res_grad_reshape_reshape)(512, 9, 9)
node meta : Tensor[34896](div_res_reshape_sequence_mask_reshape_softmax_res)(512, 9, 9)
grad meta : Tensor[34897](div_res_reshape_sequence_mask_reshape_softmax_res_grad)(512, 9, 9)