#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

#include <cmath>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <Eigen/Eigen>

template <typename Function>
class neural_network {
 public:
  using activator = Function;
  using value_type = float;
  using Matrix = Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>;
  using Vector = Eigen::Matrix<value_type, Eigen::Dynamic, 1>;

  neural_network() = default;
  neural_network(std::initializer_list<int> list) : layer_size_(list) {
    initialize();
  }
  template <typename InputIt>
  neural_network(InputIt first, InputIt last) : layer_size_(first, last) {
    initialize();
  }
  template <typename Container>
  neural_network(const Container& c) : layer_size_(c.begin(), c.end()) {
    initialize();
  }

  auto layer_count() const noexcept { return layer_size_.size(); }
  const auto& layer_size() const noexcept { return layer_size_; }
  const auto& weight() const noexcept { return weight_; }
  decltype(auto) weight(int index) const noexcept { return weight_[index]; }
  const auto& bias() const noexcept { return bias_; }
  decltype(auto) bias(int index) const noexcept { return bias_[index]; }

  neural_network& weight(int index, std::initializer_list<value_type> list) {
    assert(std::size(list) == weight_[index].size());
    auto it = std::begin(list);
    for (auto i = 0; i < std::size(list); ++i, ++it)
      weight_[index].data()[i] = *it;
    return *this;
  }

  neural_network& bias(int index, std::initializer_list<value_type> list) {
    assert(std::size(list) == bias_[index].size());
    auto it = std::begin(list);
    for (auto i = 0; i < std::size(list); ++i, ++it)
      bias_[index].data()[i] = *it;
    return *this;
  }

  template <typename InputIt>
  neural_network& weight(int index, InputIt first, InputIt last) {
    assert(weight_[index].size() == std::distance(first, last));
    auto it = first;
    for (auto i = 0; i < weight_[index].size(); ++i, ++it)
      weight_[index].data()[i] = *it;
    return *this;
  }

  template <typename InputIt>
  neural_network& bias(int index, InputIt first, InputIt last) {
    assert(bias_[index].size() == std::distance(first, last));
    auto it = first;
    for (auto i = 0; i < bias_[index].size(); ++i, ++it)
      bias_[index].data()[i] = *it;
    return *this;
  }

  template <typename Container>
  neural_network& weight(int index, const Container& c) {
    return weight(index, c.begin(), c.end());
  }

  template <typename Container>
  neural_network& bias(int index, const Container& c) {
    return bias(index, c.begin(), c.end());
  }

  const auto& input() const noexcept { return input_; }
  decltype(auto) input(int index) const noexcept { return input_[index]; }
  const auto& label() const noexcept { return label_; }
  decltype(auto) label(int index) const noexcept { return label_[index]; }
  const auto& output() const noexcept { return output_; }
  decltype(auto) output(int index) const noexcept { return output_[index]; }

  auto error() const noexcept { return error_; }

  auto& learn_rate() noexcept { return learn_rate_; }
  const auto& learn_rate() const noexcept { return learn_rate_; }
  neural_network& learn_rate(value_type new_rate) noexcept {
    learn_rate_ = new_rate;
    return *this;
  }

 private:
  activator activation{};
  std::vector<int> layer_size_{0};
  std::vector<Matrix> weight_{};
  std::vector<Vector> bias_{};

  std::vector<Vector> layer_in_{};
  std::vector<Vector> layer_out_{};
  std::vector<Matrix> weight_gradient_{};
  std::vector<Vector> bias_gradient_{};
  std::vector<Vector> gradient_tmp_{};

  std::vector<Vector> input_{};
  std::vector<Vector> label_{};
  std::vector<Vector> output_{};

  float learn_rate_{0.5};
  float error_{};

  void initialize() {
    weight_.resize(layer_count() - 1);
    bias_.resize(layer_count() - 1);

    layer_in_.resize(layer_count() - 1);
    layer_out_.resize(layer_count() - 1);

    weight_gradient_.resize(layer_count() - 1);
    bias_gradient_.resize(layer_count() - 1);
    gradient_tmp_.resize(layer_count() - 1);

    for (auto i = 0; i < layer_count() - 1; ++i) {
      weight_[i] = Matrix(layer_size_[i + 1], layer_size_[i]);
      bias_[i] = Vector(layer_size_[i + 1]);
    }
  }

  void forward_feed(const Vector& input) {
    layer_in_[0] = weight_[0] * input + bias_[0];
    layer_out_[0] = layer_in_[0].unaryExpr(activation);
    for (auto i = 1; i < layer_count() - 1; ++i) {
      layer_in_[i] = weight_[i] * layer_out_[i - 1] + bias_[i];
      layer_out_[i] = layer_in_[i].unaryExpr(activation);
    }
  }

  void backward_feed(const Vector& input, const Vector& label) {
    gradient_tmp_.back() =
        (label - layer_out_.back()).array() *
        layer_in_.back().array().unaryExpr(nabla(activation));

    for (auto i = layer_count() - 2; i > 0; --i) {
      gradient_tmp_[i - 1] =
          (weight_[i].transpose() * gradient_tmp_[i]).array() *
          layer_in_[i - 1].array().unaryExpr(nabla(activation));

      // negative gradient
      weight_gradient_[i] += gradient_tmp_[i] * layer_out_[i - 1].transpose();
      bias_gradient_[i] += gradient_tmp_[i];
    }

    // negative gradient
    weight_gradient_[0] += gradient_tmp_[0] * input.transpose();
    bias_gradient_[0] += gradient_tmp_[0];
  }

 public:
  neural_network& training_data(
      std::initializer_list<std::pair<std::initializer_list<value_type>,
                                      std::initializer_list<value_type>>>
          list) {
    input_.resize(std::size(list));
    label_.resize(std::size(list));
    auto it = std::begin(list);
    for (auto i = 0; it != std::end(list); ++i, ++it) {
      assert(std::size(it->first) == layer_size_.front());
      assert(std::size(it->second) == layer_size_.back());
      {
        input_[i] = Vector(std::size(it->first));
        auto tmp_it = std::begin(it->first);
        for (auto j = 0; tmp_it != std::end(it->first); ++j, ++tmp_it)
          input_[i].data()[j] = *tmp_it;
      }
      {
        label_[i] = Vector(std::size(it->second));
        auto tmp_it = std::begin(it->second);
        for (auto j = 0; tmp_it != std::end(it->second); ++j, ++tmp_it)
          label_[i].data()[j] = *tmp_it;
      }
    }
    return *this;
  }

  neural_network& train() {
    for (auto i = 0; i < layer_count() - 1; ++i) {
      weight_gradient_[i] = Matrix::Zero(weight_[i].rows(), weight_[i].cols());
      bias_gradient_[i] = Vector::Zero(bias_[i].size());
    }

    const float inverse_count = 1.0f / input_.size();
    // error_ = 0.0f;

    for (auto i = 0; i < input_.size(); ++i) {
      forward_feed(input_[i]);
      // error_ += 0.5 * (label_[i] - layer_out_[layer_count() -
      // 2]).squaredNorm();
      backward_feed(input_[i], label_[i]);
    }

    // error_ *= inverse_count;

    for (auto i = 0; i < layer_count() - 1; ++i) {
      weight_[i] += learn_rate_ * weight_gradient_[i] * inverse_count;
      bias_[i] += learn_rate_ * bias_gradient_[i] * inverse_count;
    }

    return *this;
  }

  neural_network& compute_output_and_error() {
    output_.resize(input_.size());
    error_ = 0.0f;
    for (auto i = 0; i < input_.size(); ++i) {
      forward_feed(input_[i]);
      output_[i] = layer_out_.back();
      error_ += 0.5 * (label_[i] - layer_out_.back()).squaredNorm();
    }
    error_ /= input_.size();
    return *this;
  }
};

template <typename Function>
std::ostream& operator<<(std::ostream& os,
                         const neural_network<Function>& network) {
  using namespace std;
  os << "----------------------------------------------------------" << endl
     << "neural network:" << endl
     << "----------------------------------------------------------" << endl;

  for (auto i = 0; i < network.layer_count() - 1; ++i) {
    os << "weight[" << i << "] = \n"
       << network.weight(i) << "\n\nbias[" << i << "] = \n"
       << network.bias(i) << "\n\n";
  }

  os << "training data with current output:" << endl;
  for (auto i = 0; i < network.input().size(); ++i) {
    os << network.input(i).transpose() << "\t\t" << network.label(i) << "\t\t"
       << network.output(i) << endl;
  }
  os << endl
     << "error = " << network.error() << endl
     << "learning rate = " << network.learn_rate() << endl
     << "----------------------------------------------------------" << endl;
  return os;
}

#endif  // NEURAL_NETWORK_H_