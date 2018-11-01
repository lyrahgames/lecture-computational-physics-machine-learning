#ifndef POLYNOMIAL_H_
#define POLYNOMIAL_H_

template <typename T>
struct Polynomial {
  using Vector = T;
  using value_type = typename T::value_type;

  Polynomial() = default;
  Polynomial(const Vector& coeffs) : parameter{coeffs} {}

  value_type operator()(value_type x) const {
    value_type result = parameter[parameter.size() - 1];
    for (int j = parameter.size() - 2; j >= 0; --j) {
      result *= x;
      result += parameter[j];
    }
    return result;
  }

  Vector parameter;
};

#endif  // POLYNOMIAL_H_