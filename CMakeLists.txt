cmake_minimum_required(VERSION 3.22.1)

project(CNN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-frtti)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization system)
find_package(Python3 COMPONENTS Development NumPy REQUIRED)


# Chercher les headers Python
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)


include_directories(
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${PYTHON_INCLUDE_DIR}
    ${Python3_NumPy_INCLUDE_DIRS}
)

add_subdirectory(src)
add_subdirectory(test)