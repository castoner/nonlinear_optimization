cmake_minimum_required(VERSION 3.0)
project(nonlinear_optimizer)

SET(CMAKE_BUILD_TYPE Release)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if(NOT Opencv_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif()
endif()

# G2O不是常用库，需要额外配置FindG2O.cmake
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
find_package(Eigen3 REQUIRED)
find_package(G2O REQUIRED)
find_package(CSparse REQUIRED)

include_directories(
        ${EIGEN3_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CSPARSE_INCLUDE_DIRS})

add_executable(g2o src/g2o_keyframe_pose_optimizer.cc)
target_link_libraries(g2o
        ${OpenCV_LIBS}
        ${CSPARSE_LIBRARY}
        g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)
