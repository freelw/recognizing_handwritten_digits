# 基于动态图的 lstm 语言模型


### forward 代码片段

backward自动计算，无需显示实现

```
for (auto input : inputs) {
    Node *i = (*(Wxi->at(input)) + Whi->at(hidden))->expand_add(Bi);
    Node *f = (*(Wxf->at(input)) + Whf->at(hidden))->expand_add(Bf);
    Node *o = (*(Wxo->at(input)) + Who->at(hidden))->expand_add(Bo);
    Node *c = (*(Wxc->at(input)) + Whc->at(hidden))->expand_add(Bc);
    Node *f_sigmoid = f->Sigmoid();
    Node *i_sigmoid = i->Sigmoid();
    Node *o_sigmoid = o->Sigmoid();
    Node *c_tanh = c->Tanh();
    cell = (*(*f_sigmoid * cell) + (*i_sigmoid * c_tanh));
    Node *cell_tanh = cell->Tanh();
    hidden = *o_sigmoid * cell_tanh;
    res.push_back(std::make_pair(hidden, cell));
}
```

### 执行效果 epochs=50

```
epoch 36 loss : 1.8984090089798 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250313_010504_37.bin
epoch 37 loss : 1.8906941413879 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250313_010504_38.bin
epoch 38 loss : 1.8831676244736 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250313_010504_39.bin
epoch 39 loss : 1.8758996725082 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250313_010504_40.bin
epoch 40 loss : 1.8687198162079 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250313_010504_41.bin
epoch 41 loss : 1.8616261482239 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250313_010504_42.bin
epoch 42 loss : 1.8547880649567 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250313_010504_43.bin
epoch 43 loss : 1.8481254577637 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250313_010504_44.bin
epoch 44 loss : 1.8414995670319 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250313_010504_45.bin
epoch 45 loss : 1.835063457489 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250313_010504_46.bin
epoch 46 loss : 1.8286859989166 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250313_010504_47.bin
epoch 47 loss : 1.8225619792938 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250313_010504_48.bin
epoch 48 loss : 1.8166818618774 emit_clip : 0
[173396/173396]checkpoint saved : ./checkpoints/checkpoint_20250313_010504_49.bin
epoch 49 loss : 1.8110052347183 emit_clip : 0
prefix : time traveller
predicted :  the strain the sart and the s
prefix : the time machine
predicted :  and the sare the sare the sar
prefix : expounding a recondite
predicted :  the strain the sart and the s
prefix :  traveller for so
predicted : ut and the sare the sare the s
prefix : it has
predicted :  a dark and the sare the sare
prefix : so most people
predicted :  the sare the sare the sare th
prefix : is simply
predicted : and the sare the sare the sare
prefix :  we cannot move about
predicted :  the strain the sart and the s
prefix : and the still
predicted :  the sare the sare the sare th

real    61m21.934s
user    422m11.407s
sys     7m34.778s
```