#include <cmath>
#include <iostream>

#include <QApplication>

#include "plot.h"

template <typename Real>
Real sigmoid(Real x) {
  return Real{1} / (Real{1} + std::exp(-x));
}

template <typename Real>
Real sigmoid_derivative(Real x) {
  return sigmoid(x) * (Real{1} - sigmoid(x));
}

template <typename Real>
Real square(Real x) {
  return x * x;
}

template <typename Real>
Real sigtanh(Real x) {
  return (Real{1} + std::tanh(x)) / Real{2};
}

template <typename Real>
Real sigtanh_derivative(Real x) {
  return (Real{1} - square(std::tanh(x))) / Real{2};
}

int main(int argc, char* argv[]) {
  QApplication application(argc, argv);
  Plot plot;

  plot.plot_function("sigmoid", sigmoid<double>, -5, 5);
  plot.plot_function("sigmoid'", sigmoid_derivative<double>, -5, 5);
  plot.plot_function("sigtanh", sigtanh<double>, -5, 5);
  plot.plot_function("sigtanh'", sigtanh_derivative<double>, -5, 5);

  return application.exec();
}