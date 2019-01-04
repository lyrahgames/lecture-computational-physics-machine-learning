cmake_minimum_required(VERSION 3.10)
project(exercise-05 VERSION 0.1.0 LANGUAGES CXX)

find_package(Eigen3 REQUIRED)
find_package(Qt5 COMPONENTS Widgets REQUIRED)

add_executable(activation activation.cc plot.h plot.cc)
target_compile_features(activation PRIVATE cxx_std_17)
target_link_libraries(activation PRIVATE Qt5::CustomPlot Eigen3::Eigen)
set_target_properties(activation PROPERTIES AUTOMOC TRUE)

add_library(neural_network INTERFACE)
target_include_directories(neural_network INTERFACE .)
target_compile_features(neural_network INTERFACE cxx_std_17)
target_link_libraries(neural_network INTERFACE Eigen3::Eigen)
add_library(neural_network::neural_network ALIAS neural_network)

add_executable(and_sigmoid network_and_sigmoid.cc)
target_link_libraries(and_sigmoid PRIVATE neural_network::neural_network)

add_executable(and_sigtanh network_and_sigtanh.cc)
target_link_libraries(and_sigtanh PRIVATE neural_network::neural_network)

add_executable(or_sigtanh network_or_sigtanh.cc)
target_link_libraries(or_sigtanh PRIVATE neural_network::neural_network)

add_executable(xor_sigtanh network_xor_sigtanh.cc)
target_link_libraries(xor_sigtanh PRIVATE neural_network::neural_network)

add_executable(nand_sigtanh network_nand_sigtanh.cc)
target_link_libraries(nand_sigtanh PRIVATE neural_network::neural_network)

add_executable(and_xor_sigtanh network_and_xor_sigtanh.cc)
target_link_libraries(and_xor_sigtanh PRIVATE neural_network::neural_network)

add_executable(and_learn_rate_sigtanh network_and_learn_rate_sigtanh.cc)
target_link_libraries(and_learn_rate_sigtanh PRIVATE neural_network::neural_network)

enable_testing()
add_subdirectory(tests)