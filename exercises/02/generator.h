#ifndef GENERATOR_H_
#define GENERATOR_H_

#include <iterator>
#include <random>

class Generator {
 public:
  Generator() = default;

  template <typename Function, typename Iterator>
  Generator& generate(const Function& f, Iterator x_begin, Iterator x_end,
                      Iterator y_begin,
                      typename Iterator::value_type std_deviation = 0,
                      typename Iterator::value_type min = 0,
                      typename Iterator::value_type max = 1) {
    using Real = typename Iterator::value_type;

    std::normal_distribution<Real> error{0, std_deviation};
    std::uniform_real_distribution<Real> distribution{min, max};
    const int size = std::distance(x_begin, x_end);
    Iterator x_it = x_begin;
    Iterator y_it = y_begin;
    for (int i = 0; i < size; ++i, ++x_it, ++y_it) {
      *x_it = distribution(rng);
      *y_it = f(*x_it) + error(rng);
    }
    return *this;
  }

 private:
  std::mt19937 rng{std::random_device{}()};
};

#endif  // GENERATOR_H_