//
// Created by pol on 11/8/19.
//

#include "Tape.h"
#include "Tensor.h"


autodiff::Variable autodiff::Tape::variable(float value, std::string const &name) {
    return autodiff::Variable(value, name, this, push_leaf());
}

autodiff::Tensor autodiff::Tape::tensor1d(std::vector<float> values, std::string const &name) {
    std::vector<uint> shape(1, values.size());
    autodiff::Tensor tensor(shape, this);
    for(int i = 0; i < values.size(); ++i){
        tensor.set_var(i, Tape::variable(values[i], name));
    }
    return tensor;
}


uint autodiff::Tape::push_leaf() {
    uint size = tape.size();
    Node x = Node({0.0, 0.0},
                  {size, size});
    tape.push_back(x);
    return size;
}

uint autodiff::Tape::push_unary(uint child, float weight) {
    uint size = tape.size();
    Node x = Node({weight, 0.0},
                  {child, size});
    tape.push_back(x);
    return size;
}



uint autodiff::Tape::push_binary(autodiff::Node var) {
    uint size = tape.size();
    tape.push_back(var);
    return size;
}

uint autodiff::Tape::size() const {
    return tape.size();
}

autodiff::Node autodiff::Tape::get_node(int i) const {
    return tape[i];
}

void autodiff::Tape::clean() {
    tape = std::vector<Node>();
}
