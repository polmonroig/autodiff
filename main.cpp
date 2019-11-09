#include <iostream>
#include <cmath>

#include "autodiff/Variable.h"


int main() {

    autodiff::Tape gradient_tape;

    auto weight = gradient_tape.variable(-0.5, "weight");
    auto bias = gradient_tape.variable(0.5, "bias");
    auto x = autodiff::Variable(1);

    auto learning_rate = autodiff::Variable(0.01);
    auto y_true = autodiff::Variable(3);
    uint n_epochs = 500;
    std::cout << "weight: " << weight.get_value() << std::endl;
    for(uint i = 0; i < n_epochs; ++i){
        std::cout << "Epoch: " << i << std::endl;
        auto y_pred = weight * x + bias;

        auto mse = (y_pred - y_true).pow(2);
        auto gradients = mse.grad();

        std::cout << "y_pred_value: " << y_pred.get_value() << std::endl;
        std::cout << "y_true_value: " << y_true.get_value() << std::endl;
        std::cout << "weight_grad: " << gradients.at(weight.get_index()).get_value() << std::endl;
        std::cout << "bias_grad: " << gradients.at(bias.get_index()).get_value() << std::endl;
        gradient_tape.clean(); // remove old nodes
        weight = weight - learning_rate * gradients.at(weight.get_index());
        bias = bias - learning_rate * gradients.at(bias.get_index());
        std::cout << "new_weight: " << weight.get_value() << std::endl;
        std::cout << "new_bias: " << bias.get_value() << std::endl;
        std::cout << "========================" << std::endl;
    }






    return 0;
}