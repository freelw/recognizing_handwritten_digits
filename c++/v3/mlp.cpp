#include "mlp.h"




Matrix *allocMatrix(Shape shape) {
    return new Matrix(shape);
}

namespace autograd {


MLP::MLP(uint _input, const std::vector<uint> &_outputs) {

    // W1 = new Parameters(allocMatrix(Shape(_outputs[0], _input)));
    // b1 = new Parameters(allocMatrix(Shape(_outputs[0], 1)));
    // W2 = new Parameters(allocMatrix(Shape(_outputs[1], _outputs[0])));
    // b2 = new Parameters(allocMatrix(Shape(_outputs[1], 1)));
}

} // namespace autograd