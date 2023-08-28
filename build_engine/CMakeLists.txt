cmake_minimum_required(VERSION 2.6)

project(yolov5)

add_definitions(-std=c++11)  # 用 C++11 标准进行编译
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)  # 不使用 CUDA 静态链接库
set(CMAKE_CXX_STANDARD 11)   # c++ 11
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

if(WIN32)
enable_language(CUDA)
endif(WIN32)

include_directories(${PROJECT_SOURCE_DIR})  # 用于添加项目源代码目录到头文件搜索路径中
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/local/TensorRT/include/)
link_directories(/usr/local/TensorRT/targets/x86_64-linux-gnu/lib/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)  # 在系统中查找 OpenCV 库，并设置 OpenCV_INCLUDE_DIRS 变量为包含 OpenCV 头文件的目录，该变量后续被用于指定项目的头文件搜索路径
include_directories(${OpenCV_INCLUDE_DIRS})  # 将 OpenCV 头文件目录添加到头文件搜索路径

# 表示编译 calibrator.cpp 和 yolov5.cpp 两个文件为名为 yolov5 的可执行文件
add_executable(yolov5 ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/yolov5.cpp)
# 链接 yolov5 可执行文件所需的四个库文件：NVIDIA TensorRT 运行库、CUDA 运行时库、myplugins 共享库和 OpenCV 库
target_link_libraries(yolov5 nvinfer)
target_link_libraries(yolov5 cudart)
target_link_libraries(yolov5 myplugins)
target_link_libraries(yolov5 ${OpenCV_LIBS})

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)

