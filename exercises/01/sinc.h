#ifndef SINC_H_
#define SINC_H_

#include <cmath>

template <typename T>
inline T sinc(T x) {
  return (x == 0) ? (1) : (std::sin(x) / x);
}

template <typename T>
inline T sinc_gradient(T x) {
  return (x == 0) ? (0) : ((x * std::cos(x) - std::sin(x)) / (x * x));
}

template <typename T>
inline T sinc_stencil(T x, T step_size) {
  return (sinc(x + step_size) - sinc(x - step_size)) / (2 * step_size);
}

#endif  // SINC_H_