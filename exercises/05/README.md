# Exercise 5

## Activation Functions
The following plot shows the two given activation functions.
They are looking very similar to each other and differ only in their growth rate.
Based on the given discrete learning problems it seems to be a good idea to choose the function `sigtanh` as the activation function, because it approximates the Heaviside step function much better.
With this choice I would expect a faster learning procedure.

Code:
- [utils.h](utils.h)
- [activation.cc](activation.cc)
- ([plot.h](plot.h) and [plot.cc](plot.cc))

![activation functions](screenshots/activation_functions.png)

## Programming

Library Code:
- [neural_network.h](neural_network.h)
- [utils.h](utils.h)

Executable Code:
- [network_and_sigtanh.cc](network_and_sigtanh.cc)
- [network_or_sigtanh.cc](network_or_sigtanh.cc)
- [network_xor_sigtanh.cc](network_xor_sigtanh.cc)
- [network_nand_sigtanh.cc](network_nand_sigtanh.cc)

Results:
- [results/and_sigtanh.txt](results/and_sigtanh.txt)
- [results/or_sigtanh.txt](results/or_sigtanh.txt)
- [results/xor_sigtanh.txt](results/xor_sigtanh.txt)
- [results/nand_sigtanh.txt](results/nand_sigtanh.txt)

In every example the error was decreasing.

## Further Analysis
### 1
There are only slight differences in the actual results.
Because the optimization algorithm works through an iterative approach the training process and the results depend on the initial values.
Therefore most of the time these values have to be chosen carefully.
This explains the differences one can see.
But we are lucky because in our case the changing of initial values did not really affect the learning procedure.

Executable Code:
- [network_xor_sigtanh.cc](network_xor_sigtanh.cc)
- [network_and_xor_sigtanh.cc](network_and_xor_sigtanh.cc)

Results:
- [results/xor_sigtanh.txt](results/xor_sigtanh.txt)
- [results/and_xor_sigtanh.txt](results/and_xor_sigtanh.txt)

### 2
Choosing a different activation function really changes the results.
`sigtanh` seems to be better suited to the problem.
See the solutions to exercise 1 above.

Executable Code:
- [network_and_sigmoid.cc](network_and_sigmoid.cc)
- [network_and_sigtanh.cc](network_and_sigtanh.cc)

Results:
- [results/and_sigmoid.txt](results/and_sigmoid.txt)
- [results/and_sigtanh.txt](results/and_sigtanh.txt)

### 3
By playing around with the learning rate I could find out that 21.1 seems to be the best value.
At first I have tried to choose a smaller learning rate.
This resulted in slightly bigger error.
Therefore I doubled the learning rate until the error exploded.
After that I have used the approach of nested intervals to find 21.1.

Executable Code:
- [network_and_learn_rate_sigtanh.cc](network_and_learn_rate_sigtanh.cc)

Results:
- [results/and_learn_rate_sigtanh.txt](results/and_learn_rate_sigtanh.txt)

## Remark
Everything was compiled with CMake.