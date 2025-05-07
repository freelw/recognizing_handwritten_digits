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
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/v5_tensor$ time ./mnist_cuda -e 10
images magic : 2051
label magic : 2049
lables_num : 60000
data loaded.
[60000/60000]loss : 0.511462
[60000/60000]loss : 0.255441
[60000/60000]loss : 0.205254
[60000/60000]loss : 0.172996
[60000/60000]loss : 0.151014
[60000/60000]loss : 0.134955
[60000/60000]loss : 0.122417
[60000/60000]loss : 0.112537
[60000/60000]loss : 0.104287
[60000/60000]loss : 0.0972957

real    0m44.968s
user    0m44.864s
sys     0m0.080s
```