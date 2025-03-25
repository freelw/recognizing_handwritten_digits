# 各种导数

## 交叉熵

### forward 

$crossentropy=-\log\frac{e^{Z_{target}-max({{Z}_i})}}{\sum_{i=1}^n e^{Z_i-max({{Z}_i})}}$

### backward

$\frac{\partial L}{\partial {Z}_i}$
$\frac{\partial L}{\partial {Z}_{target}}$