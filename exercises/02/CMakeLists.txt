cmake_minimum_required(VERSION 3.10)
project(polynomial_regression VERSION 0.1.0 LANGUAGES CXX)

find_package(Eigen3 REQUIRED)
find_package(Qt5 COMPONENTS Widgets REQUIRED)

add_executable(regression main.cc)
target_compile_features(regression PRIVATE cxx_std_11)
target_link_libraries(regression PRIVATE Qt5::CustomPlot Eigen3::Eigen)

add_executable(data_in_model_class data_in_model_class.cc plot.h plot.cc)
target_compile_features(data_in_model_class PRIVATE cxx_std_17)
target_link_libraries(data_in_model_class PRIVATE Qt5::CustomPlot Eigen3::Eigen)
set_target_properties(data_in_model_class PROPERTIES AUTOMOC TRUE)

add_executable(data_not_in_model_class data_not_in_model_class.cc plot.h plot.cc)
target_compile_features(data_not_in_model_class PRIVATE cxx_std_17)
target_link_libraries(data_not_in_model_class PRIVATE Qt5::CustomPlot Eigen3::Eigen)
set_target_properties(data_not_in_model_class PROPERTIES AUTOMOC TRUE)