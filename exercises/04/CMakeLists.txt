cmake_minimum_required(VERSION 3.10)
project(exercise-03 VERSION 0.1.0 LANGUAGES CXX)

find_package(Eigen3 REQUIRED)

add_executable(main main.cc)
target_compile_features(main PRIVATE cxx_std_17)
target_link_libraries(main PRIVATE Eigen3::Eigen)