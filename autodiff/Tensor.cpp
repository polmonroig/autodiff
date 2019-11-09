//
// Created by pol on 11/8/19.
//

#include <stdexcept>
#include <random>
#include "Tensor.h"

autodiff::Tensor::Tensor(std::vector<uint> const& shape, Tape* tape) {
    if(shape.size() > 1)throw std::logic_error("Not implemented");
    this->shape = shape;
    this->size = 1;
    this->tape = tape;
    for(unsigned int i : shape){
        this->size *= i;
    }
    this->parameters = std::vector<Variable>(size);
}

void autodiff::Tensor::set_var(int i, const autodiff::Variable &variable) {
    parameters[i] = variable;
}

autodiff::Variable autodiff::Tensor::dot(const autodiff::Tensor &t2) {
    auto out = tape->variable(0.0, "dot_out");
    for(int i = 0; i < size; ++i){
        out = out + parameters[i] * t2.parameters[i];
    }
    return out;
}

autodiff::Variable autodiff::Tensor::at(int i) {
    return parameters[i];
}

autodiff::Tensor autodiff::Tensor::operator*(const autodiff::Variable &v2) {
    auto out = tape->tensor1d(std::vector<float>(size, 0.0), "product_out");
    for(int i = 0; i < size; ++i){
        out.parameters[i] = parameters[i] * v2;
    }
    return out;
}

autodiff::Tensor autodiff::Tensor::operator+(const autodiff::Tensor &t2) {
    auto out = tape->tensor1d(std::vector<float>(size, 0.0), "product_out");
    for(int i = 0; i < size; ++i){
        out.parameters[i] = parameters[i] + t2.parameters[i];
    }
    return out;
}

autodiff::Tensor autodiff::Tensor::operator-(const autodiff::Tensor &t2) {
    auto out = tape->tensor1d(std::vector<float>(size, 0.0), "product_out");
    for(int i = 0; i < size; ++i){
        out.parameters[i] = parameters[i] - t2.parameters[i];
    }
    return out;
}

void autodiff::Tensor::add(int i, const autodiff::Variable &variable) {
    parameters[i] = parameters[i] + variable;
}

autodiff::Tensor::Tensor(std::vector<float> const &values) {
    this->shape = std::vector<uint>(1, values.size());
    this->tape = nullptr;
    this->size = 1;
    for(unsigned int i : shape){
        this->size *= i;
    }
    this->parameters = std::vector<Variable>(size);
    for(int i = 0; i < values.size(); ++i){
        set_var(i, Variable(values[i], ""));
    }

}

bool autodiff::Tensor::requires_grad() const {
    return tape != nullptr;
}


