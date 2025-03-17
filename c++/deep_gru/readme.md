# deep gru

### dropout = 0

```
base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/deep_gru$ ./deep_gru -e 0 -c ./checkpoints/checkpoint_20250317_183919_999.bin
OMP_THREADS: 8
epochs : 0
train by ../../resources/timemachine_preprocessed.txt
epochs : 0
Data loaded
loading from checkpoint : ./checkpoints/checkpoint_20250317_183919_999.bin
loaded from checkpoint
serving mode
prefix : time traveller
predicted : for so it will be 
prefix : the time machine
predicted : by h g wells i 
prefix : expounding a recondite
predicted : and and and and and 
prefix :  traveller for so
predicted : it will be convenient to 
prefix : it has
predicted : will be be convenient to 
prefix : so most people
predicted : traveller for it will be 
prefix : is simply 
predicted : traveller for it will be 
prefix :  we cannot move about
predicted : traveller for it be convenient 
prefix : and the still
predicted : to to to and and 
prefix : ask you to accept anything
predicted : to to to to and 
prefix : the time
predicted : traveller for so it will 
prefix : the time machine by h g wells i the time traveller for so it will be convenient to speak of him was expounding a recondite matter to us his grey eyes shone
predicted : and and and and and 
```

### dropout=0.2

```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/deep_gru$ ./deep_gru -e 0 -c ./checkpoints/checkpoint_20250317_191633_499.bin
OMP_THREADS: 8
epochs : 0
train by ../../resources/timemachine_preprocessed.txt
epochs : 0
Data loaded
loading from checkpoint : ./checkpoints/checkpoint_20250317_191633_499.bin
loaded from checkpoint
serving mode
prefix : time traveller
predicted : for so it will be 
prefix : the time machine
predicted : by h g wells i 
prefix : expounding a recondite
predicted : matter recondite matter recondite matter 
prefix :  traveller for so
predicted : it will be convenient to 
prefix : it has
predicted : be convenient to speak of 
prefix : so most people
predicted : wells wells i the time 
prefix : is simply 
predicted : traveller for so it will 
prefix :  we cannot move about
predicted : i the time traveller for 
prefix : and the still
predicted : be convenient to speak of 
prefix : ask you to accept anything
predicted : g wells i the time 
prefix : the time
predicted : traveller for so it will 
prefix : the time machine by h g wells i the time traveller for so it will be convenient to speak of him was expounding a recondite matter to us his grey eyes shone
predicted : and twinkled and his recondite 
```