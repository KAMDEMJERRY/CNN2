add_library(cnn_lib
    ConvLayer.cpp
    ConvLayer3D.cpp
    DenseLayer.cpp
    CNN.cpp
)

target_link_libraries(cnn_lib
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${OPENMP_LIBRARIES}
    ${Boost_LIBRARIES}
    ${Python3_LIBRARIES}
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


