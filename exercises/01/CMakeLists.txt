cmake_minimum_required(VERSION 3.10)
project(exercise_01 VERSION 0.1.0 LANGUAGES CXX)

add_subdirectory(QCustomPlot)

add_executable(exercise_01 main.cc)
target_link_libraries(exercise_01 Qt5::CustomPlot)