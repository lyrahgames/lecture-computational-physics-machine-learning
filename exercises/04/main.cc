#include <Eigen/Eigen>
#include <algorithm>
#include <cmath>
#include <iostream>

template <typename Real>
Real sigmoid(Real x) {
  return Real{1} / (Real{1} + std::exp(-x));
}

template <typename Real>
Real sigmoid_derivative(Real x) {
  return sigmoid(x) * (Real{1} - sigmoid(x));
}

template <int Layers>
struct neural_network {
  using Matrix = Eigen::MatrixXf;
  using Vector = Eigen::VectorXf;

  static constexpr auto depth() { return Layers; }

  Matrix weight[depth()];
  Vector bias[depth()];
  Vector t[depth()];
  Vector f[depth()];
  float learn_rate{0.5};
  Vector z[depth()];

  bool consistent() {
    for (auto i = 1; i < depth(); ++i) {
      if (weight[i].cols() != weight[i - 1].rows()) return false;
    }
    return true;
  }

  void forward_feed(const Vector& input) {
    t[0] = weight[0] * input + bias[0];
    f[0] = t[0].unaryExpr(std::ref(sigmoid<float>));
    for (auto i = 1; i < depth(); ++i) {
      t[i] = weight[i] * f[i - 1] + bias[i];
      f[i] = t[i].unaryExpr(std::ref(sigmoid<float>));
    }
  }

  float error(const Vector& label) {
    return 0.5 * (label - f[1]).squaredNorm();
  }

  void backward_feed(const Vector& input, const Vector& label) {
    z[depth() - 1] =
        (label - f[depth() - 1]).array() *
        t[depth() - 1].array().unaryExpr(std::ref(sigmoid_derivative<float>));

    for (auto i = depth() - 1; i > 0; --i) {
      z[i - 1] =
          (weight[i].transpose() * z[i]).array() *
          t[i - 1].array().unaryExpr(std::ref(sigmoid_derivative<float>));
      weight[i] += learn_rate * z[i] * f[i - 1].transpose();
      // bias[i] += learn_rate * z[i];
    }

    weight[0] += learn_rate * z[0] * input.transpose();
    // bias[0] += learn_rate * z[0];
  }
};

int main() {
  using namespace std;
  using namespace Eigen;

  neural_network<2> network;

  // training sample
  VectorXf input(2);
  input << 0.05, 0.10;
  VectorXf label(2);
  label << 0.01, 0.99;

  // initial weight and bias
  network.weight[0] = MatrixXf(2, 2);
  network.weight[0] << 0.15, 0.20, 0.25, 0.30;
  network.bias[0] = VectorXf(2);
  network.bias[0] << 0.35, 0.35;

  network.weight[1] = MatrixXf(2, 2);
  network.weight[1] << 0.40, 0.45, 0.50, 0.55;
  network.bias[1] = VectorXf(2);
  network.bias[1] << 0.60, 0.60;

  cout << "consistent = " << boolalpha << network.consistent() << endl;

  network.forward_feed(input);
  cout << "forward feed" << endl
       << "t[0] = \n"
       << network.t[0] << endl
       << "f[0] = \n"
       << network.f[0] << endl
       << "t[1] = \n"
       << network.t[1] << endl
       << "f[1] = \n"
       << network.f[1] << endl
       << "error = " << network.error(label) << endl
       << endl;
  network.backward_feed(input, label);
  cout << "backward feed" << endl
       << "weight[0] = \n"
       << network.weight[0] << endl
       << "bias[0] = \n"
       << network.bias[0] << endl
       << "weight[1] = \n"
       << network.weight[1] << endl
       << "bias[1] = \n"
       << network.bias[1] << endl
       << endl;
  network.forward_feed(input);
  cout << "forward feed" << endl
       << "t[0] = \n"
       << network.t[0] << endl
       << "f[0] = \n"
       << network.f[0] << endl
       << "t[1] = \n"
       << network.t[1] << endl
       << "f[1] = \n"
       << network.f[1] << endl
       << "error = " << network.error(label) << endl
       << endl;

  for (auto i = 0; i < 100; ++i) {
    network.backward_feed(input, label);
    network.forward_feed(input);
    cout << i + 2 << ": error = " << network.error(label) << endl;
  }
}