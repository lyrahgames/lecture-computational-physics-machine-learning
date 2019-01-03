#include <doctest/doctest.h>

#include <neural_network.h>
#include <utils.h>

TEST_CASE("The neural network") { neural_network<sigmoid<float>> network{}; }