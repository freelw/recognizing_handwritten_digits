# RNN 语言模型

### pytorch 的输出结果

```
(d2l) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/rnn$ python test.py 
...
epoch :  24  loss :  1.268025336302514
epoch  25  started.
epoch :  25  loss :  1.2534421859384794
epoch  26  started.
epoch :  26  loss :  1.2397092377418668
epoch  27  started.
epoch :  27  loss :  1.2341045734692153
epoch  28  started.
epoch :  28  loss :  1.2073389064460023
epoch  29  started.
epoch :  29  loss :  1.212782827349819
prefix :  time traveller
predict :  time traveller  fire fole
```

## c++ 输出

```
(d2l) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/rnn$ ./rnn 
Data loaded
...
epoch 72 loss : 1.24034
epoch 73 loss : 1.23723
epoch 74 loss : 1.23774
epoch 75 loss : 1.24489
epoch 76 loss : 1.23621
epoch 77 loss : 1.24271
epoch 78 loss : 1.2323
epoch 79 loss : 1.22673
epoch 80 loss : 1.22402
epoch 81 loss : 1.23094
epoch 82 loss : 1.22879
epoch 83 loss : 1.22603
epoch 84 loss : 1.22558
epoch 85 loss : 1.22415
epoch 86 loss : 1.22525
epoch 87 loss : 1.22706
epoch 88 loss : 1.22089
epoch 89 loss : 1.22229
epoch 90 loss : 1.22718
epoch 91 loss : 1.22483
epoch 92 loss : 1.22193
epoch 93 loss : 1.21572
epoch 94 loss : 1.20685
epoch 95 loss : 1.20439
epoch 96 loss : 1.1995
epoch 97 loss : 1.20624
epoch 98 loss : 1.20275
epoch 99 loss : 1.20292
prefix : time traveller
predicted :  and thich
prefix : the time machine
predicted :  that is f
```