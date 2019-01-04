#include <iostream>
#include <sstream>

#include <neural_network.h>
#include <utils.h>

int main(int argc, char** argv) {
  if (2 != argc) {
    std::cout << "usage: " << argv[0] << " <learn_rate>\n";
    return -1;
  }

  std::stringstream input{argv[1]};
  float learn_rate;
  input >> learn_rate;

  neural_network<sigtanh<float>> network{2, 3, 1};
  network.weight(0, {0.15, 0.20, 0.25, 0.30, 0.40, 0.45})
      .weight(1, {0.50, 0.55, 0.35})
      .bias(0, {0.60, 0.60, 0.60})
      .bias(1, {0.05})
      .training_data(
          {{{0, 0}, {0}}, {{1, 0}, {0}}, {{0, 1}, {0}}, {{1, 1}, {1}}})
      .learn_rate(learn_rate);

  std::cout << "initial network:\n" << network << std::endl;
  for (auto i = 0; i < 10000; ++i) network.train();
  std::cout << "final network:\n" << network << std::endl;
}