# RNN 语言模型

### pytorch 的输出结果

```
(d2l) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/rnn$ python test.py 
epoch  0  started.
epoch :  0  loss :  2.7649640951837813
epoch  1  started.
epoch :  1  loss :  2.0780704589117143
epoch  2  started.
epoch :  2  loss :  1.6597530912785303
epoch  3  started.
epoch :  3  loss :  1.3652211264485405
epoch  4  started.
epoch :  4  loss :  1.1150665850866408
epoch  5  started.
epoch :  5  loss :  0.8850068027774493
epoch  6  started.
epoch :  6  loss :  0.7078528120404198
epoch  7  started.
epoch :  7  loss :  0.5825978830634129
epoch  8  started.
epoch :  8  loss :  0.4883173979109242
epoch  9  started.
epoch :  9  loss :  0.4154307137997377
prefix :  time traveller
predict :  time traveller  for so is
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