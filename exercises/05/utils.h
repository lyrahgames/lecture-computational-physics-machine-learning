#ifndef UTILS_H_
#define UTILS_H_

#include <cmath>
#include <functional>

template <typename Real>
constexpr Real square(Real x) noexcept {
  return x * x;
}

template <typename Real>
struct sigmoid {
  constexpr Real operator()(Real x) const noexcept {
    return Real{1} / (Real{1} + std::exp(-x));
  }
  constexpr Real gradient(Real x) const noexcept {
    const auto tmp = operator()(x);
    return tmp * (Real{1} - tmp);
  }
};

template <typename Real>
struct sigtanh {
  constexpr Real operator()(Real x) const noexcept {
    return (Real{1} + std::tanh(x)) / Real{2};
  }
  constexpr Real gradient(Real x) const noexcept {
    return (Real{1} - square(std::tanh(x))) / Real{2};
  }
};

template <typename Function>
constexpr auto nabla(const Function& f) noexcept {
  // return std::bind(&Function::gradient, std::forward<Function>(f),
  //                  std::placeholders::_1);
  return [f](auto x) { return f.gradient(x); };
}

#endif  // UTILS_H_