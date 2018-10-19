#include <array>
#include <fstream>
#include <iostream>
#include <vector>

#include <QApplication>
#include <QCustomPlot>

#include "sinc.h"

int main(int argc, char** argv) {
  using Real = double;

  constexpr Real range_max = 3 * M_PI;
  constexpr Real range_min = -range_max;
  constexpr Real range_length = range_max - range_min;
  constexpr int step_size_count = 3;

  // We do not use the correct step sizes because
  // otherwise there is no difference in the graphs.
  std::array<Real, step_size_count> step_sizes = {2.0, 0.5, 0.001};
  std::array<int, step_size_count> sample_counts;

  // Generate sample counts and adjust the step sizes.
  for (int i = 0; i < step_size_count; ++i) {
    sample_counts[i] = 1 + range_length / step_sizes[i];
    step_sizes[i] = range_length / (sample_counts[i] - 1);
  }

  // Compute the actual data.
  std::array<std::vector<Real>, step_size_count> x_data;
  std::array<std::vector<Real>, step_size_count> y_data;
  for (int i = 0; i < step_size_count; ++i) {
    x_data[i].resize(sample_counts[i]);
    y_data[i].resize(sample_counts[i]);
    for (int j = 0; j < sample_counts[i]; ++j) {
      x_data[i][j] = range_min + j * step_sizes[i];
      y_data[i][j] = sinc(x_data[i][j]);
    }
  }

  // If there was a file path given as command line argument
  // save the generated in the given file if possible.
  if (2 == argc) {
    std::fstream file{argv[1], std::ios::out};
    if (!file.is_open()) {
      std::cout << "File could not be opened!\n";
      return -1;
    }

    for (int i = 0; i < step_size_count; ++i) {
      file << "# step size = " << step_sizes[i] << "\n";
      for (int j = 0; j < sample_counts[i]; ++j) {
        file << x_data[i][j] << "\t" << y_data[i][j] << "\n";
      }
      file << "\n";
    }
  } else {
    std::cout << "Data will not be saved!" << std::endl
              << "For saving data into a file:" << std::endl
              << argv[0] << " <file path>" << std::endl;
  }

  // Plot the data with Qt and QCustomPlot.
  QApplication application(argc, argv);
  QCustomPlot* plot = new QCustomPlot;

  plot->setWindowTitle("Plot of function 'sinc'");
  plot->xAxis->setLabel("x");
  plot->yAxis->setLabel("sinc(x)");

  std::array<QPen, step_size_count> pens = {QPen{Qt::black}, QPen{Qt::red},
                                            QPen{Qt::blue}};

  for (int i = 0; i < step_size_count; ++i) {
    plot->addGraph();
    plot->graph()->setPen(pens[i]);
    plot->graph()->setData(QVector<double>::fromStdVector(x_data[i]),
                           QVector<double>::fromStdVector(y_data[i]));
  }

  plot->rescaleAxes();
  plot->replot();
  plot->show();

  return application.exec();
}