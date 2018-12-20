cmake_minimum_required(VERSION 3.10)
project(exercise-05 VERSION 0.1.0 LANGUAGES CXX)

find_package(Eigen3 REQUIRED)
find_package(Qt5 COMPONENTS Widgets REQUIRED)

add_executable(activation activation.cc plot.h plot.cc)
target_compile_features(activation PRIVATE cxx_std_17)
target_link_libraries(activation PRIVATE Qt5::CustomPlot Eigen3::Eigen)
set_target_properties(activation PROPERTIES AUTOMOC TRUE)

add_executable(neural_network_and neural_network_and.cc)
target_compile_features(neural_network_and PRIVATE cxx_std_17)
target_link_libraries(neural_network_and PRIVATE Eigen3::Eigen)