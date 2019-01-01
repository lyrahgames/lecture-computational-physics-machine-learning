#include <QApplication>

#include "plot.h"
#include "utils.h"

int main(int argc, char* argv[]) {
  QApplication application(argc, argv);

  Plot plot{};
  plot.plot_function("sigmoid", sigmoid<double>{}, -5, 5)
      .plot_function("sigmoid'", nabla(sigmoid<double>{}), -5, 5)
      .plot_function("sigtanh", sigtanh<double>{}, -5, 5)
      .plot_function("sigtanh'", nabla(sigtanh<double>{}), -5, 5);

  return application.exec();
}