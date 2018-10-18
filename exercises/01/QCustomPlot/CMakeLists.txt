find_package(Qt5 COMPONENTS Widgets PrintSupport REQUIRED)

add_library(qcustomplot qcustomplot.cpp qcustomplot.h)
target_link_libraries(qcustomplot PUBLIC Qt5::Widgets Qt5::PrintSupport)
target_compile_features(qcustomplot PUBLIC cxx_std_11)
target_include_directories(qcustomplot PUBLIC .)
set_target_properties(qcustomplot PROPERTIES AUTOMOC TRUE)

add_library(Qt5::CustomPlot ALIAS qcustomplot)