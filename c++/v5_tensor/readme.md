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
epoch : 0 [50000/50000] loss : 0.671061
evaluating :  [10000/10000] correct : 9157
epoch : 1 [50000/50000] loss : 0.30326
evaluating :  [10000/10000] correct : 9279
epoch : 2 [50000/50000] loss : 0.262538
evaluating :  [10000/10000] correct : 9372
epoch : 3 [50000/50000] loss : 0.234863
evaluating :  [10000/10000] correct : 9450
epoch : 4 [50000/50000] loss : 0.212027
evaluating :  [10000/10000] correct : 9468
epoch : 5 [50000/50000] loss : 0.192673
evaluating :  [10000/10000] correct : 9501
epoch : 6 [50000/50000] loss : 0.17642
evaluating :  [10000/10000] correct : 9545
epoch : 7 [50000/50000] loss : 0.162765
evaluating :  [10000/10000] correct : 9569
epoch : 8 [50000/50000] loss : 0.151023
evaluating :  [10000/10000] correct : 9586
epoch : 9 [50000/50000] loss : 0.140758
evaluating :  [10000/10000] correct : 9597

real    0m34.149s
user    0m34.023s
sys     0m0.113s
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
epoch : 0 [50000/50000] loss : 0.707109
evaluating :  [10000/10000] correct : 9072
epoch : 1 [50000/50000] loss : 0.333386
evaluating :  [10000/10000] correct : 9203
epoch : 2 [50000/50000] loss : 0.30727
evaluating :  [10000/10000] correct : 9247
epoch : 3 [50000/50000] loss : 0.296601
evaluating :  [10000/10000] correct : 9286
epoch : 4 [50000/50000] loss : 0.289447
evaluating :  [10000/10000] correct : 9322
epoch : 5 [50000/50000] loss : 0.285547
evaluating :  [10000/10000] correct : 9349
epoch : 6 [50000/50000] loss : 0.286223
evaluating :  [10000/10000] correct : 9370
epoch : 7 [50000/50000] loss : 0.289685
evaluating :  [10000/10000] correct : 9392
epoch : 8 [50000/50000] loss : 0.295575
evaluating :  [10000/10000] correct : 9403
epoch : 9 [50000/50000] loss : 0.303859
evaluating :  [10000/10000] correct : 9413

real    0m2.858s
user    0m2.436s
sys     0m0.387s
```