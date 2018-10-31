#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <Eigen/Eigen>

#include <QApplication>
#include <QCustomPlot>

using Real = double;
using Matrix =
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

template <typename Real>
Real function(Real x) {
  return 2 * x;
}

int main(int argc, char** argv) {
  constexpr int training_sample_count = 1000;
  constexpr int plot_sample_count = 1000;
  constexpr Real plot_min = 0;
  constexpr Real plot_max = 1;
  constexpr Real plot_step_size =
      (plot_max - plot_min) / (plot_sample_count - 1);
  constexpr Real std_deviation = 0.3;

  std::mt19937 rng{std::random_device{}()};
  std::normal_distribution<Real> error{0, std_deviation};
  std::uniform_real_distribution<Real> training_distribution{0, 1};

  std::vector<double> x_data(training_sample_count);
  std::vector<double> y_data(training_sample_count);

  // Construct data with random errors.
  for (int i = 0; i < training_sample_count; ++i) {
    x_data[i] = training_distribution(rng);
    y_data[i] = function(x_data[i]) + error(rng);
  }

  // Learn parameters for different model classes.
  std::vector<Vector> parameters;
  for (auto order : {1, 3, 10}) {
    // Build system matrix for polynomial regression.
    Matrix design_matrix(training_sample_count, order + 1);
    for (int i = 0; i < design_matrix.rows(); ++i) {
      design_matrix(i, 0) = 1;
      for (int j = 1; j < design_matrix.cols(); ++j)
        design_matrix(i, j) = design_matrix(i, j - 1) * x_data[i];
    }

    // Solve the system of linear equations with
    // Pivot-Householder-QR-decomposition.
    Eigen::Map<Vector> rhs(y_data.data(), training_sample_count);
    parameters.push_back(design_matrix.colPivHouseholderQr().solve(rhs));
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

  std::vector<double> plot_x_data(plot_sample_count);
  for (int i = 0; i < plot_sample_count; ++i) {
    plot_x_data[i] = plot_min + i * plot_step_size;
  }

  for (auto parameter : parameters) {
    std::vector<double> estimated_data(plot_sample_count);
    for (int i = 0; i < plot_sample_count; ++i) {
      estimated_data[i] = parameter[parameter.size() - 1];
      for (int j = parameter.size() - 2; j >= 0; --j) {
        estimated_data[i] *= plot_x_data[i];
        estimated_data[i] += parameter[j];
      }
    }

    plot->addGraph();
    plot->graph()->setPen(QPen{Qt::blue, 2});
    plot->graph()->setData(QVector<double>::fromStdVector(plot_x_data),
                           QVector<double>::fromStdVector(estimated_data));
  }

  plot->rescaleAxes();
  plot->replot();
  plot->show();

  return application.exec();
}