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
      .learn_rate(0.5);

  std::cout << "AND initial network:\n" << network << std::endl;
  for (auto i = 0; i < 10000; ++i) network.train();
  std::cout << "AND final network:\n" << network << std::endl;
  network.training_data(
      {{{0, 0}, {0}}, {{1, 0}, {1}}, {{0, 1}, {1}}, {{1, 1}, {0}}});
  std::cout << "XOR initial network:\n" << network << std::endl;
  for (auto i = 0; i < 10000; ++i) network.train();
  std::cout << "XOR final network:\n" << network << std::endl;
}