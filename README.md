# autodiff #
Reverse Mode Automatic Differentiation 

Even though there are lots of automatic differentiation libraries and deep learning frameworks,
in order to gain knowledge in the inner process of these frameworks I made a personal implementation 
of reverse mode automatic differentiation in C++.

## How it works? ##
Like most implementations out there, this is based on a gradient tape that stores a implicit graph 
dynamically and updates it every time a new operation takes place. Here is note on what each class does. 

### Tape ### 
When first importing the library into your code a new tape will be initialized, note that even you can create a new tape, only the 
initial one will be useful for calculations. This tape will be initialized under the namespace autodiff 
with the name gradient_tape. The only time you need to refer to that variable is when you want to 
remove old calculations and empty the tape, using the gradient_tape.clean() method.

### Variable ### 
This class represents a new variable in a calculation, if you want to track its gradient you need to 
set the require_grad flag to true otherwise it will be treated as a mathematical constant. You can 
perform all kinds of operations with these variables and the tape will take care of calculating their value, 
finally when you want to make the backward pass to generate the gradients you should call the grad method. 
This will return a tensor with the gradients of all the variables that have been tracked

### Tensor ###
This class simply represents a mathematical Tensor, that stores Variables, they can either require gradient 
or not. You can perform different operations with them, such as dot product and matrix multiplication.
 

## Example ##
The following code represents a sample that simulates a neural network with two layers, no bias, 
and trains 100 epochs using gradient descent, applying the simple perceptron update rule. It also 
applies MSE. 
```cpp
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
    auto y_pred = x.matmul(weights).matmul(weights2);
    auto tmp = (y_pred - y);
    auto mse = tmp.pow(2).mean();
    auto gradients = mse.grad();
    autodiff::gradient_tape.clean();
    weights.apply_gradients(learning_rate, gradients);
    weights2.apply_gradients(learning_rate, gradients);
}
```