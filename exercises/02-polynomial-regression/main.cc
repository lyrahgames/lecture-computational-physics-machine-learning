#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <Eigen/Eigen>

#include <QApplication>
#include <QCustomPlot>

template <typename Real>
Real first_function(Real x) {
  return 2 * x * x;
}

template <typename Real>
Real second_function(Real x) {
  return 2 * x - 10 * std::pow(x, 5) + 15 * std::pow(x, 10);
}

int main(int argc, char** argv) {
  using Real = double;
  constexpr int sample_count = 1000;
  constexpr Real std_deviation{1};
  constexpr auto function = second_function<Real>;
  constexpr int order = 20;

  std::mt19937 rng{std::random_device{}()};
  std::normal_distribution<Real> error{0, std_deviation};
  std::uniform_real_distribution<Real> x_distribution{-1, 1};

  std::vector<double> x_data(sample_count);
  std::vector<double> y_data(sample_count);

  // Construct data with random errors.
  for (int i = 0; i < sample_count; ++i) {
    x_data[i] = x_distribution(rng);
    y_data[i] = function(x_data[i]) + error(rng);
  }

  // Build system matrix for polynomial regression.
  using Matrix =
      Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
  Matrix system_matrix(sample_count, order + 1);
  for (int i = 0; i < system_matrix.rows(); ++i) {
    system_matrix(i, 0) = 1;
    for (int j = 1; j < system_matrix.cols(); ++j) {
      system_matrix(i, j) = system_matrix(i, j - 1) * x_data[i];
    }
  }

  Eigen::Map<Vector> rhs(y_data.data(), sample_count);
  Vector parameter = system_matrix.colPivHouseholderQr().solve(rhs);

  std::vector<double> estimated_data(sample_count);
  for (int i = 0; i < sample_count; ++i) {
    estimated_data[i] = parameter[order];
    for (int j = order - 1; j >= 0; --j) {
      estimated_data[i] *= x_data[i];
      estimated_data[i] += parameter[j];
    }
  }

  // Plot the data with Qt and QCustomPlot.
  QApplication application(argc, argv);
  QCustomPlot* plot = new QCustomPlot;

  plot->setWindowTitle("Plot");
  plot->xAxis->setLabel("x");
  plot->yAxis->setLabel("y");

  plot->addGraph();
  plot->graph()->setPen(QPen{Qt::black, 2});
  plot->graph()->setLineStyle(QCPGraph::lsNone);
  plot->graph()->setScatterStyle(QCPScatterStyle::ssCircle);
  plot->graph()->setData(QVector<double>::fromStdVector(x_data),
                         QVector<double>::fromStdVector(y_data));

  plot->addGraph();
  plot->graph()->setPen(QPen{Qt::blue, 2});
  plot->graph()->setData(QVector<double>::fromStdVector(x_data),
                         QVector<double>::fromStdVector(estimated_data));

  plot->rescaleAxes();
  plot->replot();
  plot->show();

  return application.exec();
}