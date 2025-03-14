# gru with embedding

### timemachine_middle.txt 训练1000+轮左右的效果

训练数据：通过前9个单词预测下一个

```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/gru_embedding$ ./gru -e 0 -c ./cp_middle_loss_0_35.bin 
OMP_THREADS: 8
epochs : 0
train by ../../resources/timemachine_preprocessed.txt
epochs : 0
Data loaded
loading from checkpoint : ./cp_middle_loss_0_35.bin
loaded from checkpoint
serving mode
prefix : time traveller
predicted : for so it will be 
prefix : the time machine
predicted : by h g wells i 
prefix : expounding a recondite
predicted : matter to us his grey 
prefix :  traveller for so
predicted : it will be convenient to 
prefix : it has
predicted : not that flashed and his 
prefix : so most people
predicted : sat and lazily admired his 
prefix : is simply 
predicted : they taught you at school 
prefix :  we cannot move about
predicted : us to begin upon said 
prefix : and the still
predicted : burned brightly and his usually 
prefix : ask you to accept anything
predicted : without reasonable ground for it 
prefix : the time
predicted : traveller for so it will 
prefix : the time machine by h g wells i the time traveller for so it will be convenient to speak of him was expounding a recondite matter to us his grey eyes shone
predicted : and his usually pale face 
```