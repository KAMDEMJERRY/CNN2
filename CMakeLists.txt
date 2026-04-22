cmake_minimum_required(VERSION 3.22.1)
project(CNN)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Build type ────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Type de build" FORCE)
endif()

# ── Flags communs ─────────────────────────────────────────────────────────────
add_compile_options(-frtti)
# add_compile_definitions(EIGEN_DONT_PARALLELIZE) 

# ── Flags par mode ────────────────────────────────────────────────────────────
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3 -mavx2 -mfma -march=native -DNDEBUG)
  message(STATUS "Mode RELEASE — optimisations SIMD activées")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  add_compile_options(-O2 -mavx2 -mfma -g)
  message(STATUS "Mode RELWITHDEBINFO — optimisé + symboles debug")
else()
  add_compile_options(-O0 -g -fsanitize=address)
  add_link_options(-fsanitize=address)
  message(STATUS "Mode DEBUG — ASan activé")
endif()

# ── Option benchmark (oneDNN / TBB uniquement) ────────────────────────────────
option(ENABLE_BENCHMARK "Activer le mode benchmark (oneDNN, TBB, perf)" OFF)

# ── Dépendances ───────────────────────────────────────────────────────────────
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization system)
find_package(Python3 COMPONENTS Development NumPy REQUIRED)
find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
  set(USE_OPENMP TRUE)
else()
  message(WARNING "OpenMP not found, parallelization disabled")
  set(USE_OPENMP FALSE)
endif()

if(ENABLE_BENCHMARK)
  find_package(dnnl REQUIRED)
  find_package(TBB REQUIRED)
endif()

# ── Includes ──────────────────────────────────────────────────────────────────
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
    "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
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
  ${OpenMP_CXX_INCLUDE_DIRS}
)

add_subdirectory(src)
enable_testing()
add_subdirectory(test)