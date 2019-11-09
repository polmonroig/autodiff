//
// Created by pol on 11/8/19.
//

#include "Variable.h"
#include <cmath>


autodiff::Variable::Variable(float value, std::string const &name, autodiff::Tape *tape, uint index) {
    this->value = value;
    this->name = name;
    this->tape = tape;
    this->index = index;

}
autodiff::Variable::Variable() = default;



autodiff::Tensor autodiff::Variable::grad() {

    uint size = tape->size();
    Tensor gradients = autodiff::Tensor(std::vector<float>(size));
    gradients.set_var(index, autodiff::Variable(1.0, ""));
    for(int i = size - 1; i >= 0; --i){
        Node node = tape->get_node(i);
        gradients.add(node.children.first, autodiff::Variable(node.weights.first, "") * gradients.at(i));
        if(node.children.second < size && node.children.second >= 0){
            gradients.add(node.children.second, autodiff::Variable(node.weights.second, "") * gradients.at(i));
        }

    }
    return gradients;
}

float autodiff::Variable::get_value() const{
    return value;
}


autodiff::Variable autodiff::Variable::operator*(const Variable &v2) {
    if(requires_grad() && v2.requires_grad()){
        return Variable(value * v2.value, "mul",
                        tape, tape->push_binary(Node({v2.value, value}, {index, v2.index})));
    }
    else if(requires_grad()){
        auto out = *this;
        out.value *= v2.value;
        out.index = tape->push_unary(index, out.value);
        return out;
    }
    else if(v2.requires_grad()){
        auto out = v2;
        out.value *= value;
        out.index = tape->push_unary(index, out.value);
        return out;
    }
    else{
        return Variable(value * v2.value, "mul");
    }
}


autodiff::Variable autodiff::Variable::operator+(const Variable &v2) {

    if(requires_grad() && v2.requires_grad()){
        return Variable(value + v2.value, "add",
                        tape, tape->push_binary(Node({1.0, 1.0}, {index, v2.index})));
    }
    else if(requires_grad()){
        auto out = *this;
        out.value += v2.value;
        out.index = tape->push_unary(index, out.value);
        return out;
    }
    else if(v2.requires_grad()){
        auto out = v2;
        out.value += value;
        out.index = tape->push_unary(index, out.value);
        return out;
    }
    else{
        return Variable(value + v2.value, "add");
    }

}

autodiff::Variable autodiff::Variable::operator-(const Variable &v2) {

    if(requires_grad() && v2.requires_grad()){
        return Variable(value - v2.value, "rest",
                        tape, tape->push_binary(Node({1.0, 1.0}, {index, v2.index})));
    }
    else if(requires_grad()){
        auto out = *this;
        out.value -= v2.value;
        out.index = tape->push_unary(index, out.value);
        return out;
    }
    else if(v2.requires_grad()){
        auto out = v2;
        out.value -= value;
        out.index = tape->push_unary(index, out.value);
        return out;
    }
    else{
        return Variable(value - v2.value, "rest");
    }

}



uint autodiff::Variable::get_index() const {
    return index;
}

autodiff::Variable autodiff::Variable::sin() const {
    if(requires_grad()){
        return Variable(std::sin(value), "sin", tape,
                        tape->push_unary(index, std::cos(value)));
    }
    else {
        return Variable(std::sin(value), "sin");
    }

}

autodiff::Variable autodiff::Variable::pow(float power) const {
    if(requires_grad()){
        return Variable(std::pow(value, power), "pow", tape,
                        tape->push_unary(index, power));
    }
    else{
        return Variable(std::pow(value, power), "pow");
    }

}


autodiff::Variable::Variable(float value, std::string const &name) {
    this->value = value;
    this->name = name;
}

bool autodiff::Variable::requires_grad() const {
    return tape != nullptr;
}


