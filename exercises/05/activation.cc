#include <cmath>
#include <functional>
#include <iostream>

#include <QApplication>

#include "plot.h"

template <typename Real>
Real square(Real x) {
  return x * x;
}

template <typename Real>
struct sigmoid {
  Real operator()(Real x) const noexcept {
    return Real{1} / (Real{1} + std::exp(-x));
  }
  Real gradient(Real x) const noexcept {
    const auto tmp = operator()(x);
    return tmp * (Real{1} - tmp);
  }
};

template <typename Real>
struct sigtanh {
  Real operator()(Real x) const noexcept {
    return (Real{1} + std::tanh(x)) / Real{2};
  }
  Real gradient(Real x) const noexcept {
    return (Real{1} - square(std::tanh(x))) / Real{2};
  }
};

template <typename Function>
auto nabla(Function&& f) {
  return std::bind(&Function::gradient, std::forward<Function>(f),
                   std::placeholders::_1);
}

int main(int argc, char* argv[]) {
  QApplication application(argc, argv);

  Plot plot;
  plot.plot_function("sigmoid", sigmoid<double>{}, -5, 5);
  plot.plot_function("sigmoid'", nabla(sigmoid<double>{}), -5, 5);
  plot.plot_function("sigtanh", sigtanh<double>{}, -5, 5);
  plot.plot_function("sigtanh'", nabla(sigtanh<double>{}), -5, 5);

  return application.exec();
}