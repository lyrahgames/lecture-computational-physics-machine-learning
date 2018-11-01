#ifndef POLYNOMIAL_REGRESSION_H_
#define POLYNOMIAL_REGRESSION_H_

#include <iterator>

#include <Eigen/Eigen>

template <typename Input_iterator, typename Output_iterator>
void polynomial_regression(Input_iterator x_begin, Input_iterator x_end,
                           Input_iterator y_begin,
                           Output_iterator parameter_begin,
                           Output_iterator parameter_end) {
  using Real = typename Input_iterator::value_type;
  using Matrix =
      Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

  const int sample_size = std::distance(x_begin, x_end);
  const int parameter_size = std::distance(parameter_begin, parameter_end);

  // Build system matrix for polynomial regression.
  Matrix design_matrix(sample_size, parameter_size);
  Input_iterator x_it = x_begin;
  for (int i = 0; i < design_matrix.rows(); ++i, ++x_it) {
    design_matrix(i, 0) = 1;
    for (int j = 1; j < design_matrix.cols(); ++j)
      design_matrix(i, j) = design_matrix(i, j - 1) * (*x_it);
  }

  // Solve the system of linear equations with
  // Pivot-Householder-QR-decomposition.
  // Eigen::Map<Vector> rhs(y_begin.data(), sample_size);
  Vector rhs(sample_size);
  Input_iterator y_it = y_begin;
  for (int i = 0; i < sample_size; ++i, ++y_it) {
    rhs[i] = *y_it;
  }
  Vector parameter = design_matrix.colPivHouseholderQr().solve(rhs);
  Output_iterator p_it = parameter_begin;
  for (int i = 0; i < parameter_size; ++i, ++p_it) {
    *p_it = parameter[i];
  }
}

#endif  // POLYNOMIAL_REGRESSION_H_