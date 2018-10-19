#include <iostream>
#include <sstream>
#include <vector>

#include <QApplication>
#include <QCustomPlot>

#include "sinc.h"

int main(int argc, char** argv) {
  using Real = float;

  constexpr Real range_max = 3 * M_PI;
  constexpr Real range_min = -range_max;
  constexpr Real range_length = range_max - range_min;

  int sample_count_min{500};
  int sample_count_max{10000};

  // Read minimal and maximal sample counts from command line if possible.
  // Otherwise use the default values.
  if (3 != argc) {
    std::cout << "Minimal and maximal sample count were defaulted!" << std::endl
              << "Use command line arguments for manual setting:" << std::endl
              << argv[0] << " <minimal sample count> <maximal sample count>"
              << std::endl;
  } else {
    std::stringstream input{argv[1]};
    input >> sample_count_min;

    input = std::stringstream{argv[2]};
    input >> sample_count_max;
  }

  std::cout << "min sample count = " << sample_count_min << std::endl
            << "max sample count = " << sample_count_max << std::endl;

  // Compute mean errors for different sample counts.
  // Vectors have to use double because of QCustomPlot.
  const int data_size = sample_count_max - sample_count_min + 1;
  std::vector<double> sample_counts(data_size);
  std::vector<double> step_sizes(data_size);
  std::vector<double> error_data(data_size, 0);
  for (int i = 0; i < data_size; ++i) {
    sample_counts[i] = sample_count_min + i;
    const Real step_size = range_length / (sample_counts[i] - 1);
    step_sizes[i] = step_size;
    // Choose only samples that do not lie on the boundary.
    for (int j = 1; j < sample_counts[i] - 1; ++j) {
      const Real x = range_min + j * step_size;
      error_data[i] += std::abs(sinc_gradient(x) - sinc_stencil(x, step_size));
    }
    error_data[i] /= sample_counts[i];
  }

  // Plot the data with Qt and QCustomPlot.
  QApplication application(argc, argv);

  QCustomPlot* sample_count_plot = new QCustomPlot;
  sample_count_plot->setWindowTitle("Mean error by sample count");
  sample_count_plot->xAxis->setLabel("sample count");
  sample_count_plot->yAxis->setLabel("mean error");
  sample_count_plot->addGraph();
  sample_count_plot->graph()->setData(
      QVector<double>::fromStdVector(sample_counts),
      QVector<double>::fromStdVector(error_data));
  sample_count_plot->rescaleAxes();
  sample_count_plot->replot();
  sample_count_plot->show();

  QCustomPlot* step_size_plot = new QCustomPlot;
  step_size_plot->setWindowTitle("Mean error by step size");
  step_size_plot->xAxis->setLabel("step size");
  step_size_plot->yAxis->setLabel("mean error");
  step_size_plot->addGraph();
  step_size_plot->graph()->setData(QVector<double>::fromStdVector(step_sizes),
                                   QVector<double>::fromStdVector(error_data));
  step_size_plot->rescaleAxes();
  step_size_plot->replot();
  step_size_plot->show();

  return application.exec();
}