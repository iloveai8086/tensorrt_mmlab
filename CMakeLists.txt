cmake_minimum_required(VERSION 3.19)
project(tensorrt_mmlab)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.1/bin/nvcc")
find_package(CUDA REQUIRED)

if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL  "aarch64")
    set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
    set(CUDA_INSTALL_TARGET_DIR targets/aarch64-linux)
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    set(CMAKE_C_COMPILER /usr/bin/gcc)
    set(CMAKE_CXX_COMPILER /usr/bin/g++)
    set(CUDA_INSTALL_TARGET_DIR targets/x86_64-linux)
endif()

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.1)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/${CUDA_INSTALL_TARGET_DIR}/include)
set(CMAKE_BUILD_TYPE "Debug")  # Release
set(CMAKE_CXX_FLAGS_RELEASE "-Wno-deprecated-declarations -O2")
add_compile_options(-W)
add_compile_options(-std=c++11)

set( SMS 30 32 35 37 50 52 53 60 61 62 70 72 75 87)
foreach(sm ${SMS})
    set(GENCODE ${GENCODE} -gencode arch=compute_${sm},code=sm_${sm})
endforeach()
set(HIGHEST_SM 87)
set(GENCODE ${GENCODE} -gencode arch=compute_${HIGHEST_SM},code=compute_${HIGHEST_SM})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        -ccbin ${CMAKE_CXX_COMPILER}
        -Xcompiler -DWIN_INTERFACE_CUSTOM
        -Xcompiler -I/usr/aarch64-linux-gnu/include/
        -Xlinker -lsocket
        -Xlinker -rpath=/usr/lib/aarch64-linux-gnu/
        -Xlinker -rpath=/usr/aarch64-linux-gnu/lib/
        -Xlinker -L/usr/lib/aarch64-linux-gnu/
        -Xlinker -L/usr/aarch64-linux-gnu/lib/
        )

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    message("Using Debug Mode")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G --ptxas-options=-v)
endif()

set(CUDNN_LIBRARY "/media/ros/A666B94D66B91F4D/ros/new_deploy/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/lib/libcudnn.so")
set(CUDNN_INCLUDE_DIR "/media/ros/A666B94D66B91F4D/ros/new_deploy/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/include")

if(CUDNN_LIBRARY)
    message("CUDNN is available!")
    message("CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
    set(CUDNN_AVAIL ON)
else()
    message("CUDNN is NOT Available")
    set(CUDNN_AVAIL OFF)
endif()

include_directories(
        ${CUDNN_INCLUDE_DIR}
        include
        /usr/local/cuda-11.1/include
        /media/ros/A666B94D66B91F4D/ros/new_deploy/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1/include
)

link_directories(
        /usr/local/cuda-11.1/lib64
        /media/ros/A666B94D66B91F4D/ros/new_deploy/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1/lib
        /media/ros/A666B94D66B91F4D/ros/new_deploy/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/lib
)


add_executable(tensorrt_mmlab main.cpp)
target_link_libraries(${PROJECT_NAME}
        cudart
        libnvinfer.so
        libnvonnxparser.so
        )