cmake_minimum_required(VERSION 3.10)
project(polynomial_regression VERSION 0.1.0 LANGUAGES CXX)

find_package(Eigen3 REQUIRED)
add_subdirectory(QCustomPlot)

add_executable(plot main.cc)
target_compile_features(plot PRIVATE cxx_std_11)
target_link_libraries(plot PRIVATE Qt5::CustomPlot Eigen3::Eigen)