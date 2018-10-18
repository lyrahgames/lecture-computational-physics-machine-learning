#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <QCustomPlot>
#include <QVector>

inline double sinc(double x) { return (x == 0) ? (1) : (std::sin(x) / x); }

template <typename T>
struct Sinc {
  using value_type = T;
  value_type operator()(value_type x) {
    return (x == 0) ? (1) : (std::sin(x) / x);
  }
  value_type gradient(value_type x) {
    return (x == 0) ? (0) : ((x * std::cos(x) - std::sin(x)) / (x * x));
  }
};

int main(int argc, char** argv) {
  if (2 != argc) {
    std::cout << "usage: " << argv[0] << " <step size>\n";
    return -1;
  }

  std::stringstream input{argv[1]};
  double step_size;
  input >> step_size;

  std::cout << "input step_size = " << step_size << std::endl;

  constexpr double max = 3 * M_PI;
  constexpr double min = -max;
  constexpr double length = max - min;

  int count = length / step_size;
  if (count <= 0) {
    std::cout << "Error: Sample count is to small!\n";
    return -1;
  }

  // adjust step size
  step_size = length / count;

  std::cout << "adjusted step_size = " << step_size << std::endl
            << "sample count = " << count << std::endl;

  // compute the data
  std::vector<double> x_data(count);
  std::vector<double> y_data(count);

  Sinc<double> function{};

  for (int i = 0; i < count; ++i) {
    x_data[i] = min + i * step_size;
    y_data[i] = function(x_data[i]);
  }

  const double inverse_step_size = 1 / step_size;
  std::vector<double> d_data(count);
  d_data[0] = inverse_step_size * (y_data[1] - y_data[0]);
  d_data[count - 1] =
      inverse_step_size * (y_data[count - 1] - y_data[count - 2]);
  for (int i = 1; i < count - 1; ++i) {
    d_data[i] = 0.5 * inverse_step_size * (y_data[i + 1] - y_data[i - 1]);
  }

  std::vector<double> error_data(count);
  error_data[0] = error_data[count - 1] = 0;
  for (int i = 1; i < count - 1; ++i) {
    error_data[i] = std::abs(function.gradient(x_data[i]) - d_data[i]);
  }

  // write data to file
  std::fstream file("data.txt", std::ios::out);
  if (!file.is_open()) {
    std::cout << "Error: File could not be opened!\n";
    return -1;
  }

  for (int i = 0; i < count; ++i) {
    file << x_data[i] << "\t" << y_data[i] << "\n";
  }

  QApplication application(argc, argv);
  QCustomPlot* plot = new QCustomPlot;
  plot->addGraph();
  plot->graph()->setData(QVector<double>::fromStdVector(x_data),
                         QVector<double>::fromStdVector(y_data));
  plot->addGraph();
  plot->graph()->setData(QVector<double>::fromStdVector(x_data),
                         QVector<double>::fromStdVector(d_data));
  plot->rescaleAxes();
  plot->replot();
  plot->show();

  QCustomPlot* error_plot = new QCustomPlot;
  error_plot->addGraph();
  error_plot->graph()->setData(QVector<double>::fromStdVector(x_data),
                               QVector<double>::fromStdVector(error_data));
  error_plot->rescaleAxes();
  error_plot->replot();
  error_plot->show();

  return application.exec();
}