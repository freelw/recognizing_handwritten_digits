# gru with embedding

### timemachine_middle.txt 训练1000+轮左右的效果

训练数据：通过前9个单词预测下一个

```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/gru_embedding$ ./gru -e 0 -c ./checkpoints/checkpoint_20250314_160914_851.bin
OMP_THREADS: 8
epochs : 0
train by ../../resources/timemachine_preprocessed.txt
epochs : 0
Data loaded
loading from checkpoint : ./checkpoints/checkpoint_20250314_160914_851.bin
loaded from checkpoint
serving mode
prefix : time traveller
predicted : for so it and the 
prefix : the time machine
predicted : by h wells i shall 
prefix : expounding a recondite
predicted : matter to be convenient to 
prefix :  traveller for so
predicted : it and the soft radiance 
prefix : it has
predicted : not to begin the soft 
prefix : so most people
predicted : it and the soft radiance 
prefix : is simply 
predicted : time traveller for so it 
prefix :  we cannot move about
predicted : it and the soft radiance 
prefix : and the still
predicted : burned brightly and the soft 
prefix : ask you to accept anything
predicted : without reasonable ground for it 
prefix : the time
predicted : traveller for so it and 
prefix : the time machine by h g wells i the time traveller for so it will be convenient to speak of him was expounding a recondite matter to us his grey eyes shone
predicted : and the soft radiance of 
```