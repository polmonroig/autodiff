//
// Created by pol on 11/8/19.
//

#include <stdexcept>
#include <random>
#include <chrono>
#include "Tensor.h"

autodiff::Tensor::Tensor(std::vector<uint> const& shape, bool requires_grad) {
    this->shape = shape;
    this->_requires_grad = requires_grad;
    if(shape.size() > 1){ // > 1 dimension
        this->size = 1;
        this->is_leaf = false;
        std::vector<uint> new_shape;
        for(int j = 1; j < shape.size(); ++j){
            new_shape.push_back(shape[j]);
        }
        for(int i = 0; i < shape[0]; ++i){
            this->sub_tensors.emplace_back(new_shape, requires_grad);
        }
    }
    else{ // 1 dimension
        this->is_leaf = true;
        this->size = shape[0];
        this->parameters = std::vector<Variable>(this->size);
        if(requires_grad){
            record_parameters();
        }
    }
}

bool autodiff::Tensor::requires_grad() const{
    return _requires_grad;
}

void autodiff::Tensor::record_parameters(){
    for(auto & parameter : parameters){
        parameter.record_var(gradient_tape.push_leaf());
    }
}

 autodiff::Tensor autodiff::Tensor::rand(std::vector<uint> const& shape, bool requires_grad){
    autodiff::Tensor out = autodiff::Tensor(shape, false);
    out.fill_random(requires_grad);
    return out;
}

void autodiff::Tensor::fill_random(bool requires_grad){

    if(shape.size() > 1){ // > 1 dimension
        for(int i = 0; i < shape[0]; ++i){
            sub_tensors[i].fill_random(requires_grad);
        }
    }
    else{ // 1 dimension
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator (seed);
        std::uniform_real_distribution<double> distribution(0.0,1.0);
        for(int i = 0; i < shape[0]; ++i){
            parameters[i] = Variable(distribution(generator), requires_grad);
        }
    }
}

void autodiff::Tensor::set_var(std::vector<uint> const &indexes, autodiff::Variable const& var) {
    if(indexes.size() > 1){
        autodiff::Tensor *current = &this->sub_tensors[indexes[0]];
        for(int i = 1; i < indexes.size() - 1; ++i){
            current = &current->sub_tensors[indexes[i]];
        }
        current->parameters[indexes.size() - 1] = var;
    }
    else{
        parameters[indexes[0]] = var;
    }

}

void autodiff::Tensor::add(std::vector<uint> const &indexes, autodiff::Variable const& var) {
    if(indexes.size() > 1){
        autodiff::Tensor *current = &this->sub_tensors[indexes[0]];
        for(int i = 1; i < indexes.size() - 1; ++i){
            current = &current->sub_tensors[indexes[i]];
        }
        current->parameters[indexes[indexes.size() - 1]]  = current->parameters[indexes[indexes.size() - 1]] + var;
    }
    else{
        parameters[indexes[0]] = parameters[indexes[0]] + var;
    }

}

autodiff::Variable autodiff::Tensor::dot(autodiff::Tensor const& t2){
    Variable result(0.0, requires_grad() || t2.requires_grad()) ;
    for(int i = 0; i < shape[0]; ++i){
        result = result + parameters[i] * t2.parameters[i];
    }
    return result;
}

autodiff::Tensor autodiff::Tensor::matmul(autodiff::Tensor & t2){
    auto out = Tensor({shape[0], t2.shape[1]}, requires_grad() || t2.requires_grad());
    for(uint i = 0; i < shape[0]; ++i){
        for(uint j = 0; j < t2.shape[1]; ++j){
            for(uint k = 0; k < shape[1]; ++k){
                out.add({i, j}, this->at({i, k}) * t2.at({k, j}));
            }
        }
    }

    return out;
}


autodiff::Variable autodiff::Tensor::at(std::vector<uint> const &indexes) {
    if(indexes.size() > 1){
        autodiff::Tensor *current = &this->sub_tensors[indexes[0]];
        for(int i = 1; i < indexes.size() - 1; ++i){
            current = &current->sub_tensors[indexes[i]];
        }
        return current->parameters[indexes[indexes.size() - 1]];
    }
    else{
        return parameters[indexes[0]];
    }

}

// Pre: t1 and this have the same shape
autodiff::Tensor autodiff::Tensor::operator-(autodiff::Tensor const& t2) const{
    Tensor out = Tensor(shape, this->requires_grad() || t2.requires_grad());
    if(shape.size() > 1){
        for(int i = 0; i < sub_tensors.size(); ++i){
            out.sub_tensors[i] = sub_tensors[i] - t2.sub_tensors[i];
        }
    }
    else{
        for(int i = 0; i < parameters.size(); ++i){
            out.parameters[i] =  parameters[i] - t2.parameters[i];
        }

    }
    return out;
}


autodiff::Variable autodiff::Tensor::mean(){
    Variable avg = Variable(0.0, true, "mean");
    if(shape.size() > 1){
        for(int i = 0; i < sub_tensors.size(); ++i){
            avg = avg + sub_tensors[i].mean();
        }
        avg = avg / sub_tensors.size();
    }
    else{
        for(const auto & parameter : parameters){
           avg = avg + parameter;
        }
        avg = avg / parameters.size();
    }

    return avg;
}

autodiff::Tensor autodiff::Tensor::pow(float power) const{
    Tensor out = Tensor(shape, requires_grad());
    if(shape.size() > 1){
        for(int i = 0; i < sub_tensors.size(); ++i) {
            out.sub_tensors[i] = sub_tensors[i].pow(power);
        }
    }
    else{
        for(int i = 0; i < parameters.size(); ++i){
            out.parameters[i] = parameters[i].pow(power);
        }
    }

    return out;
}

void autodiff::Tensor::apply_gradients(autodiff::Variable const& learning_rate, autodiff::Tensor const& gradients){
    if(shape.size() > 1){
        for(int i = 0; i < sub_tensors.size(); ++i){
            sub_tensors[i].apply_gradients(learning_rate, gradients);
        }

    }
    else{
        for(auto & parameter : parameters){
            parameter = parameter - learning_rate * gradients.parameters[parameter.get_index()];
        }
    }
}

autodiff::Tensor autodiff::Tensor::sigmoid(const autodiff::Tensor &tensor) {
    Tensor out = Tensor(tensor.shape, tensor.requires_grad());
    if(tensor.shape.size() > 1){
        for(int i = 0; i < tensor.sub_tensors.size(); ++i) {
            out.sub_tensors[i] = Tensor::sigmoid(tensor.sub_tensors[i]);
        }
    }
    else{
        for(int i = 0; i < tensor.parameters.size(); ++i){
            out.parameters[i] = Variable::sigmoid(tensor.parameters[i]);
        }
    }

    return out;
}
