cmake_minimum_required(VERSION 3.10)
project(polynomial_regression VERSION 0.1.0 LANGUAGES CXX)

find_package(Eigen3 REQUIRED)

add_executable(regression main.cc)
target_compile_features(regression PRIVATE cxx_std_11)
target_link_libraries(regression PRIVATE Qt5::CustomPlot Eigen3::Eigen)