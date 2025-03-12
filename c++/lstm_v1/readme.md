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