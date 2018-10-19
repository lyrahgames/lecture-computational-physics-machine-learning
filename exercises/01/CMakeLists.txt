cmake_minimum_required(VERSION 3.10)
project(exercise_01 VERSION 0.1.0 LANGUAGES CXX)

add_subdirectory(QCustomPlot)

add_executable(plot plot.cc)
target_compile_features(plot PRIVATE cxx_std_11)
target_link_libraries(plot Qt5::CustomPlot)

add_executable(error stencil_error.cc)
target_compile_features(error PRIVATE cxx_std_11)
target_link_libraries(error Qt5::CustomPlot)