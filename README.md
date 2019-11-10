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
 