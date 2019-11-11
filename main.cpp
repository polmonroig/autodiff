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
    auto y = autodiff::Tensor::rand({batch_size, 1}, false);
    auto learning_rate = autodiff::Variable(0.001);

    uint n_epochs = 100;
    for(uint i = 0; i < n_epochs; ++i){
        std::cout << "Epoch: " << i << std::endl;
        auto y_pred = x.matmul(weights).matmul(weights2);
        auto tmp = (y_pred - y);
        auto mse = tmp.pow(2).mean();
        auto gradients = mse.grad();
        autodiff::gradient_tape.clean();
        weights.apply_gradients(learning_rate, gradients);
        weights2.apply_gradients(learning_rate, gradients);
        std::cout << "Loss: " << mse.get_value() << std::endl;
        std::cout << "====================" << std::endl;
    }


    return 0;
}