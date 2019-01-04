#include <iostream>

#include <neural_network.h>
#include <utils.h>

int main() {
  neural_network<sigtanh<float>> network{2, 3, 1};
  network.weight(0, {0.15, 0.20, 0.25, 0.30, 0.40, 0.45})
      .weight(1, {0.50, 0.55, 0.35})
      .bias(0, {0.60, 0.60, 0.60})
      .bias(1, {0.05})
      .training_data(
          {{{0, 0}, {0}}, {{1, 0}, {0}}, {{0, 1}, {0}}, {{1, 1}, {1}}})
      .learn_rate(0.5)
      .compute_output_and_error();

  std::cout << network << std::endl;

  network.train().compute_output_and_error();
  std::cout << network << std::endl;
}