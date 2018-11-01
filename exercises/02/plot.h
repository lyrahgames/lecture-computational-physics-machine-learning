#ifndef PLOT_H_
#define PLOT_H_

#include <iterator>
#include <vector>

#include <QApplication>
#include <QCustomPlot>
#include <QKeyEvent>

class Plot : public QCustomPlot {
  Q_OBJECT

  static const std::vector<QColor> colors;

 public:
  Plot(QWidget* parent = nullptr) : QCustomPlot(parent) {
    setWindowTitle("Plot");
    xAxis->setLabel("x");
    yAxis->setLabel("y");

    resize(800, 450);

    show();
  }

  void next_color() { color_index = (color_index + 1) % colors.size(); }

  template <typename Function>
  Plot& plot_function(const Function& f) {
    constexpr int plot_sample_count = 1000;
    constexpr double plot_min = 0;
    constexpr double plot_max = 1;
    constexpr double plot_step_size =
        (plot_max - plot_min) / (plot_sample_count - 1);

    std::vector<double> x_data(plot_sample_count);
    std::vector<double> estimated_data(plot_sample_count);
    for (int i = 0; i < plot_sample_count; ++i) {
      x_data[i] = plot_min + i * plot_step_size;
      estimated_data[i] = f(x_data[i]);
    }

    addGraph();
    graph()->setPen(QPen{colors[color_index], 2});
    graph()->setData(QVector<double>::fromStdVector(x_data),
                     QVector<double>::fromStdVector(estimated_data));
    replot();
    rescaleAxes();
    next_color();
    return *this;
  }

  template <typename Iterator>
  Plot& plot_data(Iterator x_begin, Iterator x_end, Iterator y_begin) {
    const int size = std::distance(x_begin, x_end);
    QVector<double> x_data(size), y_data(size);
    Iterator x_it = x_begin;
    Iterator y_it = y_begin;
    for (int i = 0; i < size; ++i, ++x_it, ++y_it) {
      x_data[i] = *x_it;
      y_data[i] = *y_it;
    }

    addGraph();
    graph()->setPen(QPen{colors[color_index], 2});
    graph()->setLineStyle(QCPGraph::lsNone);
    graph()->setScatterStyle(QCPScatterStyle::ssCircle);
    graph()->setData(x_data, y_data);

    replot();
    rescaleAxes();
    next_color();
    return *this;
  }

 protected:
  void keyPressEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_Escape) QApplication::quit();
  }

 private:
  int color_index = 0;
};

#endif  // PLOT_H_