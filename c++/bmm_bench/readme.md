# vs pytorch bench

```
(d2l) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/bmm_bench$ time ./bmm_bench 

real    0m51.284s
user    6m28.135s
sys     0m1.421s

(d2l) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/bmm_bench$ time python bench.py 

real    0m3.568s
user    0m15.805s
sys     0m0.887s
```

20250321 优化
```
(d2l) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/bmm_bench$ time ./bmm_bench 

real    0m19.235s
user    2m31.721s
sys     0m0.948s
```