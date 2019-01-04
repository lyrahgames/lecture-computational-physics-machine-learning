#include <Eigen/Eigen>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

template <typename Real>
Real sigmoid(Real x) {
  return Real{1} / (Real{1} + std::exp(-x));
}

template <typename Real>
Real sigmoid_derivative(Real x) {
  return sigmoid(x) * (Real{1} - sigmoid(x));
}

template <typename Real>
Real square(Real x) {
  return x * x;
}

template <typename Real>
Real sigtanh(Real x) {
  return (Real{1} + std::tanh(x)) / Real{2};
}

template <typename Real>
Real sigtanh_derivative(Real x) {
  return (Real{1} - square(std::tanh(x))) / Real{2};
}

template <int Layers>
struct neural_network {
  using Matrix = Eigen::MatrixXf;
  using Vector = Eigen::VectorXf;

  static constexpr auto depth() { return Layers; }

  Matrix weight[depth()];
  Matrix weight_gradient[depth()];
  Vector bias[depth()];
  Vector bias_gradient[depth()];
  Vector t[depth()];
  Vector f[depth()];
  float learn_rate{0.5};
  Vector z[depth()];
  float error_;

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
    return 0.5 * (label - f[depth() - 1]).squaredNorm();
  }

  float error() const { return error_; }

  void backward_feed(const Vector& input, const Vector& label) {
    z[depth() - 1] =
        (label - f[depth() - 1]).array() *
        t[depth() - 1].array().unaryExpr(std::ref(sigmoid_derivative<float>));

    for (auto i = depth() - 1; i > 0; --i) {
      z[i - 1] =
          (weight[i].transpose() * z[i]).array() *
          t[i - 1].array().unaryExpr(std::ref(sigmoid_derivative<float>));
      weight_gradient[i] += z[i] * f[i - 1].transpose();
      bias_gradient[i] += z[i];
    }

    // negative gradient
    weight_gradient[0] += z[0] * input.transpose();
    bias_gradient[0] += z[0];
  }

  template <typename Iterator>
  void learn(Iterator input_first, Iterator input_last, Iterator label_first) {
    for (auto i = 0; i < depth(); ++i) {
      weight_gradient[i] = Matrix::Zero(weight[i].rows(), weight[i].cols());
      bias_gradient[i] = Vector::Zero(bias[i].size());
    }

    const auto count = distance(input_first, input_last);
    const float inverse_count = 1.0f / count;

    error_ = 0.0f;
    for (auto it = input_first, label_it = label_first; it != input_last;
         ++it, ++label_it) {
      forward_feed(*it);
      error_ += error(*label_it);
      backward_feed(*it, *label_it);
    }
    error_ *= inverse_count;

    for (auto i = 0; i < depth(); ++i) {
      weight[i] += learn_rate * weight_gradient[i] * inverse_count;
      bias[i] += learn_rate * bias_gradient[i] * inverse_count;
    }
  }

  template <typename Input_iterator, typename Output_iterator>
  void compute(Input_iterator input_first, Input_iterator input_last,
               Output_iterator output_first) {
    auto it = input_first;
    auto out_it = output_first;
    for (; it != input_last; ++it, ++out_it) {
      forward_feed(*it);
      *out_it = f[depth() - 1];
    }
  }
};

int main() {
  using namespace std;
  using namespace Eigen;

  neural_network<2> network;

  // training sample
  // vector<VectorXf> inputs{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  vector<VectorXf> inputs{VectorXf(2), VectorXf(2), VectorXf(2), VectorXf(2)};
  inputs[0] << 0, 0;
  inputs[1] << 0, 1;
  inputs[2] << 1, 0;
  inputs[3] << 1, 1;
  vector<VectorXf> labels{VectorXf::Zero(1), VectorXf::Zero(1),
                          VectorXf::Zero(1), VectorXf::Ones(1)};
  vector<VectorXf> outputs(inputs.size());

  // initial weight and bias
  network.weight[0] = MatrixXf(3, 2);
  network.weight[0] << 0.15, 0.20, 0.25, 0.30, 0.40, 0.45;
  network.bias[0] = VectorXf(3);
  network.bias[0] << 0.60, 0.60, 0.60;

  network.weight[1] = MatrixXf(1, 3);
  network.weight[1] << 0.50, 0.55, 0.35;
  network.bias[1] = VectorXf(1);
  network.bias[1] << 0.05;

  cout << "consistent = " << boolalpha << network.consistent() << endl;

  cout << "initial:" << endl
       << "weight[0] = \n"
       << network.weight[0] << endl
       << "bias[0] = \n"
       << network.bias[0] << endl
       << "weight[1] = \n"
       << network.weight[1] << endl
       << "bias[1] = \n"
       << network.bias[1] << endl
       << "error = " << network.error() << endl
       << endl;

  network.compute(begin(inputs), end(inputs), begin(outputs));
  for (auto i = 0; i < inputs.size(); ++i) {
    cout << inputs[i].transpose() << endl
         << labels[i] << "\t" << outputs[i] << endl;
  }
  cout << endl;

  const int iterations = 1;
  for (auto i = 0; i < iterations; ++i) {
    network.learn(begin(inputs), end(inputs), begin(labels));
  }
  cout << "iterations = " << iterations << ":" << endl
       << "weight[0] = \n"
       << network.weight[0] << endl
       << "bias[0] = \n"
       << network.bias[0] << endl
       << "weight[1] = \n"
       << network.weight[1] << endl
       << "bias[1] = \n"
       << network.bias[1] << endl
       << "error = " << network.error() << endl
       << endl;

  network.compute(begin(inputs), end(inputs), begin(outputs));
  for (auto i = 0; i < inputs.size(); ++i) {
    cout << inputs[i].transpose() << endl
         << labels[i] << "\t" << outputs[i] << endl;
  }
  cout << endl;
}