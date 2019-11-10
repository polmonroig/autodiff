#include <iostream>
#include "autodiff/Variable.h"
#include "autodiff/Tensor.h"

int main() {

    uint n_neurons = 10;
    uint n_inputs = 1;
    uint batch_size = 10;
    auto weights = autodiff::Tensor::rand({n_inputs, n_neurons}, true);
    auto weights2 = autodiff::Tensor::rand({n_neurons,  1}, true);
    auto x = autodiff::Tensor::rand({batch_size, n_inputs}, false);
    auto y = autodiff::Tensor::rand({batch_size, n_neurons}, false);
    auto learning_rate = autodiff::Variable(0.001);
    auto y_pred = x.matmul(weights).matmul(weights2);


    /*auto w = autodiff::Variable(3.0, true);
    auto x = autodiff::Variable(3.0);
    auto y = autodiff::Variable(3.0);
    auto learning_rate = autodiff::Variable(0.001);
    uint n_epochs = 100;
    std::cout << "weight: " << w.get_value() << std::endl;
    for(uint i = 0; i < n_epochs; ++i){
        std::cout << "Epoch: " << i << std::endl;
        auto y_pred = x*w;
        auto mse = (y_pred - y).pow(2);
        auto gradients = mse.grad();
        std::cout << "y_pred: " << y_pred.get_value() << std::endl;
        std::cout << "y_true: " << y.get_value() << std::endl;
        std::cout << "w_grad: " << gradients.at({w.get_index()}).get_value() << std::endl;
        autodiff::gradient_tape.clean();
        w = w - learning_rate * gradients.at({w.get_index()});


        std::cout << "new_weight: " << w.get_value() << std::endl;
        std::cout << "====================" << std::endl;
    }*/

    return 0;
}