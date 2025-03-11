# 模拟forward负载

### OMP_THREADS=4

```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/matrix_bench$ time ./bench 
OMP_THREADS : 4
epoch : 0 i : 169

real    1m43.441s
user    6m39.531s
sys     0m2.177s
```

### OMP_THREADS=8

```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/matrix_bench$ time ./bench 
OMP_THREADS : 8
epoch : 0 i : 169

real    1m44.501s
user    6m42.902s
sys     0m2.057s
```