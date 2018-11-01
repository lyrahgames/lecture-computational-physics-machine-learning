#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

#include <QApplication>

#include "generator.h"
#include "plot.h"
#include "polynomial.h"
#include "polynomial_regression.h"

template <typename Real>
Real function(Real x) {
  return 2 * x - 10 * std::pow(x, 5) + 15 * std::pow(x, 10);
}

int main(int argc, char** argv) {
  using Real = double;

  // Read values from command line or set them to default values.
  int sample_count{100};
  Real std_deviation{0.3};
  int test_sample_count{0};
  if (argc > 1) {
    std::stringstream input{argv[1]};
    input >> sample_count;
    if (argc > 2) {
      input = std::stringstream{argv[2]};
      input >> std_deviation;
      if (argc > 3) {
        input = std::stringstream{argv[3]};
        input >> test_sample_count;
      }
    }
  } else {
    std::cout
        << "Default parameters are used.\nusage: " << argv[0]
        << " [<sample count> [<standard deviation> [<test sample count>]]]\n";
  }

  QApplication application(argc, argv);
  Plot plot;

  std::vector<Real> x_data(sample_count);
  std::vector<Real> y_data(sample_count);

  Generator generator;
  generator.generate(function<float>, x_data.begin(), x_data.end(),
                     y_data.begin(), std_deviation);
  plot.plot_data(x_data.begin(), x_data.end(), y_data.begin());

  for (auto order : {1, 3, 10}) {
    std::vector<Real> parameter(order + 1);
    polynomial_regression(x_data.begin(), x_data.end(), y_data.begin(),
                          parameter.begin(), parameter.end());
    plot.plot_function(Polynomial{parameter});
  }

  return application.exec();
}