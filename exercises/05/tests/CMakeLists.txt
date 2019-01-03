find_package(doctest REQUIRED)

add_executable(main_test
  doctest_main.cc
  neural_network.cc
)
target_link_libraries(main_test
  PRIVATE
    doctest::doctest
    neural_network::neural_network
)

add_test(main_test main_test)