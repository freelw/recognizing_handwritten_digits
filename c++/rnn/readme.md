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
epoch 0 loss : 2.74366
epoch 1 loss : 2.14464
epoch 2 loss : 1.80121
epoch 3 loss : 1.51176
epoch 4 loss : 1.2766
epoch 5 loss : 1.08683
epoch 6 loss : 0.919671
epoch 7 loss : 0.764669
epoch 8 loss : 0.635798
epoch 9 loss : 0.538493
prefix : time traveller
predicted :  spone and
prefix : the time machine
predicted :  by hie  s
```