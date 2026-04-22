add_library(cnn_lib
    shared.cpp
    ConvLayer.cpp
    ConvLayerDataParallel.cpp
    ConvLayer3DDataParallel.cpp
    ConvLayer3D.cpp
    DenseLayer.cpp
    DenseLayerDataParallel.cpp
    DenseLayerModelParallel.cpp
    CNN.cpp
)

target_link_libraries(cnn_lib
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${OPENMP_LIBRARIES}
    ${Boost_LIBRARIES}
    ${Python3_LIBRARIES}
    ${OpenMP_CXX_FLAGS}
    # PRIVATE dotenv-cpp
)

# ======================================

add_executable(CNN
    main.cpp
)

target_link_libraries(CNN
    cnn_lib
)


# ======================================

add_executable(DIMS
    dimensions.cpp    
)

target_link_libraries(DIMS
    cnn_lib
)

# ======================================


add_executable(PLAY
    playground.cpp    
)

target_link_libraries(PLAY
    cnn_lib
)


macro(add_exec name main)
add_executable(${name} ${main})
target_link_libraries(${name}
    cnn_lib
)
endmacro()

add_exec(CNN2D CNN2D.cpp)