include(GoogleTest)

# ── Tests originaux ──────────────────────────────────────────────────────────

add_executable(test_Tensor test_Tensor.cpp)
target_link_libraries(test_Tensor PRIVATE GTest::GTest GTest::Main Eigen3::Eigen)
gtest_discover_tests(test_Tensor)

add_executable(test_ConvLayer test_ConvLayer.cpp ../src/ConvLayer.cpp)
target_link_libraries(test_ConvLayer PRIVATE GTest::GTest GTest::Main Eigen3::Eigen OpenMP::OpenMP_CXX)
gtest_discover_tests(test_ConvLayer)

# ── Nouveaux Tests ───────────────────────────────────────────────────────────

add_executable(test_DenseLayer test_DenseLayer.cpp ../src/DenseLayer.cpp)
target_link_libraries(test_DenseLayer PRIVATE GTest::GTest GTest::Main Eigen3::Eigen OpenMP::OpenMP_CXX)
gtest_discover_tests(test_DenseLayer)

add_executable(test_ActivationLayer test_ActivationLayer.cpp)
target_link_libraries(test_ActivationLayer PRIVATE GTest::GTest GTest::Main Eigen3::Eigen OpenMP::OpenMP_CXX)
gtest_discover_tests(test_ActivationLayer)

add_executable(test_PoolLayer test_PoolLayer.cpp)
target_link_libraries(test_PoolLayer PRIVATE GTest::GTest GTest::Main Eigen3::Eigen OpenMP::OpenMP_CXX)
gtest_discover_tests(test_PoolLayer)

add_executable(test_LossLayer test_LossLayer.cpp)
target_link_libraries(test_LossLayer PRIVATE GTest::GTest GTest::Main Eigen3::Eigen OpenMP::OpenMP_CXX)
gtest_discover_tests(test_LossLayer)

add_executable(test_Optimizer test_Optimizer.cpp)
target_link_libraries(test_Optimizer PRIVATE GTest::GTest GTest::Main Eigen3::Eigen OpenMP::OpenMP_CXX)
gtest_discover_tests(test_Optimizer)

add_executable(test_DropoutLayer test_DropoutLayer.cpp)
target_link_libraries(test_DropoutLayer PRIVATE GTest::GTest GTest::Main Eigen3::Eigen OpenMP::OpenMP_CXX)
gtest_discover_tests(test_DropoutLayer)

add_executable(test_Dimensions test_Dimensions.cpp ../src/dimensions.cpp)
target_link_libraries(test_Dimensions PRIVATE GTest::GTest GTest::Main Eigen3::Eigen OpenMP::OpenMP_CXX)
gtest_discover_tests(test_Dimensions)

add_executable(test_CNN test_CNN.cpp ../src/CNN.cpp ../src/ConvLayer.cpp ../src/ConvLayer3D.cpp ../src/DenseLayer.cpp)
target_link_libraries(test_CNN PRIVATE GTest::GTest GTest::Main Eigen3::Eigen OpenMP::OpenMP_CXX)
gtest_discover_tests(test_CNN)