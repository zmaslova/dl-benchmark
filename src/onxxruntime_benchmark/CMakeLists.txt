cmake_minimum_required(VERSION 3.13)
project(onnxruntime_benchmark)

find_package(OpenCV REQUIRED core highgui)
find_package(nlohmann_json REQUIRED)

file(GLOB sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h*")

add_executable(${PROJECT_NAME} ${sources} ${headers})

target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include" "${ONNXRuntime_DIR}/include"
    "${ONNXRuntime_DIR}/include/onnxruntime/core/session")

target_link_libraries(${PROJECT_NAME} PRIVATE opencv_core "${ONNXRuntime_DIR}/build/libonnxruntime.so")

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 14 LINKER_LANGUAGE CXX)
