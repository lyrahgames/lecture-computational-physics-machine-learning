#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

// #include <cmath>
// #include <iterator>
// #include <stdexcept>
// #include <string>
// #include <vector>

// #include <Eigen/Eigen>

// template <typename Function>
// class neural_network {
//  public:
//   using activator = Function;
//   using Matrix = Eigen::MatrixXf;
//   using Vector = Eigen::VectorXf;

//   bool consistent() const;

//   void forward_feed(const Vector& input) {
//     t[0] = weight[0] * input + bias[0];
//     f[0] = t[0].unaryExpr(std::ref(sigmoid<float>));
//     for (auto i = 1; i < depth(); ++i) {
//       t[i] = weight[i] * f[i - 1] + bias[i];
//       f[i] = t[i].unaryExpr(std::ref(sigmoid<float>));
//     }
//   }

//   auto depth() const { return depth_; }
//   auto depth(int new_depth) {
//     if (new_depth <= 0)
//       throw std::invalid_argument{
//           "Depth of neural network has to be bigger than zero!"};
//     depth_ = new_depth;

//     weight_.resize(depth_);
//     bias_.resize(depth_);

//     return *this;
//   }

//  private:
//   int depth_;
//   std::vector<Matrix> weight_;
//   std::vector<Vector> bias_;

//   std::vector<Vector> t_;
//   std::vector<Vector> f_;
//   std::vector<Matrix> weight_gradient_;
//   std::vector<Vector> bias_gradient_;
//   std::vector<Vector> z_;
//   float learn_rate_{0.5};
//   float error_{};
// };

// template <typename Function>
// bool neural_network<Function>::consistent() const {
//   if (depth_ <= 0) return false;
//   if (weight_.size() != depth_) return false;
//   if (bias_.size() != depth_) return false;
//   for (auto i = 0; i < depth_; ++i)
//     if (weight_[i].rows() != bias_[i].size()) return false;
//   for (auto i = 1; i < depth_; ++i)
//     if (weight_[i].cols() != weight_[i - 1].rows()) return false;
//   return true;
// }

template <typename Function>
class neural_network {};

#endif  // NEURAL_NETWORK_H_