//
// Created by pol on 11/8/19.
//

#include "Variable.h"
#include <cmath>


autodiff::Variable::Variable(float value, bool requires_grad, std::string const& name) {
    this->value = value;
    this->name = name;
    this->_requires_grad = requires_grad;
    if(requires_grad){
        this->index = autodiff::gradient_tape.push_leaf();
    }
}
autodiff::Variable::Variable() = default;



autodiff::Tensor autodiff::Variable::grad() {

    uint size = gradient_tape.size();
    Tensor gradients = autodiff::Tensor({size});
    gradients.set_var({index}, autodiff::Variable(1.0, false));
    for(int i = size - 1; i >= 0; --i){
        Node node = gradient_tape.get_node(i);
        gradients.add({node.children.first}, autodiff::Variable(node.weights.first, false) * gradients.at({uint(i)}));
        if(node.children.second < size && node.children.second >= 0){
            gradients.add({node.children.second}, autodiff::Variable(node.weights.second, false) * gradients.at({uint(i)}));
        }

    }
    return gradients;
}

float autodiff::Variable::get_value() const{
    return value;
}


autodiff::Variable autodiff::Variable::operator*(const Variable &v2)const {
    if(requires_grad() && v2.requires_grad()){
        auto new_var = Variable(value * v2.value, false, "mul");
        new_var.record_var(gradient_tape.push_binary(Node({v2.value, value}, {index, v2.index})));
        return new_var;

    }
    else if(requires_grad()){
        auto out = *this;
        out.value *= v2.value;
        out.index = gradient_tape.push_unary(index, out.value);
        return out;
    }
    else if(v2.requires_grad()){
        auto out = v2;
        out.value *= value;
        out.index = gradient_tape.push_unary(index, out.value);
        return out;
    }
    else{
        return Variable(value * v2.value, false, "mul");
    }
}


autodiff::Variable autodiff::Variable::operator+(const Variable &v2) const{

    if(requires_grad() && v2.requires_grad()){
        auto new_var = Variable(value + v2.value, false, "add");
        new_var.record_var(gradient_tape.push_binary(Node({1.0, 1.0}, {index, v2.index})));
        return new_var;

    }
    else if(requires_grad()){
        auto out = *this;
        out.value += v2.value;
        out.index = gradient_tape.push_unary(index, out.value);
        return out;
    }
    else if(v2.requires_grad()){
        auto out = v2;
        out.value += value;
        out.index = gradient_tape.push_unary(index, out.value);
        return out;
    }
    else{
        return Variable(value + v2.value, false, "add");
    }

}

autodiff::Variable autodiff::Variable::operator-(const Variable &v2) {

    if(requires_grad() && v2.requires_grad()){
        auto new_var = Variable(value - v2.value, false, "rest");
        new_var.record_var(gradient_tape.push_binary(Node({1.0, 1.0}, {index, v2.index})));
        return new_var;

    }
    else if(requires_grad()){
        auto out = *this;
        out.value -= v2.value;
        out.index = gradient_tape.push_unary(index, out.value);
        return out;
    }
    else if(v2.requires_grad()){
        auto out = v2;
        out.value -= value;
        out.index =  gradient_tape.push_unary(index, out.value);
        return out;
    }
    else{
        return Variable(value - v2.value, false, "rest");
    }

}





uint autodiff::Variable::get_index() const {
    return index;
}

autodiff::Variable autodiff::Variable::sin() const {
    if(requires_grad()){
        auto new_var = Variable(std::sin(value), false, "sin");
        new_var.record_var(gradient_tape.push_unary(index, std::cos(value)));
        return new_var;
    }
    else {
        return Variable(std::sin(value), false, "sin");
    }

}

autodiff::Variable autodiff::Variable::pow(float power) const {
    if(requires_grad()){
        auto new_var = Variable(std::pow(value, power), false, "pow");
        new_var.record_var(gradient_tape.push_unary(index, power));
        return new_var;
    }
    else{
        return Variable(std::pow(value, power), false, "pow");
    }

}


bool autodiff::Variable::requires_grad() const {
    return _requires_grad;
}

void autodiff::Variable::record_var(int i) {
    this->index = i;
    _requires_grad = true;
}


