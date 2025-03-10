### pytorch

```
epoch :  29  loss :  0.7979641133858785
prefix :  time traveller
predict :  time traveller  wth xpecsseddou s u
prefix :  the time machine
predict :  the time machine  put an widaca u u n
prefix :  expounding a recondite
predict :  expounding a recondite ndo  wat a mmaccesdd
prefix :   traveller for so
predict :   traveller for so  wsouskilles bbbacae
```


```
(d2l) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/lstm$ python test.py
epoch  0  started.
[ 173395 / 173396 ]epoch :  0  loss :  1.9362292463825401
prefix :  time traveller
predict :  time traveller  far far far far far
prefix :  the time machine
predict :  the time machine  wore far far far fa
prefix :  expounding a recondite
predict :  expounding a recondite  wore far far far fa
prefix :   traveller for so
predict :   traveller for so  wore far far far fa
prefix :  it has
predict :  it has  far far far far far
prefix :  so most people
predict :  so most people  far far far far far
prefix :  is simply
predict :  is simply  an he far far far fa
prefix :   we cannot move about
predict :   we cannot move about  he far far far far
prefix :  and the still
predict :  and the still  fare wore far far f
epoch  1  started.
```

### c++ predict

```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/lstm$ ./lstm -e 0 -c ./checkpoints/checkpoint_20250310_204527_24.bin
epochs : 0
train by ../../resources/timemachine_preprocessed.txt
epochs : 0
Data loaded
loading from checkpoint : ./checkpoints/checkpoint_20250310_204527_24.bin
loaded from checkpoint
serving mode
prefix : time traveller
predicted :  fonthes a to s fogeovero tave


...
prefix : the time machine
predicted :  by an an was as a ta ad an w
...

prefix : expounding a recondite
predicted :  mam t ts f mad f con wy w sat
prefix :  traveller for so
predicted :  it inthis n tnthis is n to hi
prefix : it has
predicted :  ly a pe t t te t tr y t t te
prefix : so most people
predicted : ssfos s gets ov o sf s s te sa
prefix : is simply
predicted : hasanibein oresonchischisanith
prefix :  we cannot move about
predicted : neat to ughes n tnts t th n to
prefix : and the still
predicted : ow i or has a ge mas t ie hasa
```