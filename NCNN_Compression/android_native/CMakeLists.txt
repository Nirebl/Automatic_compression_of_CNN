cmake_minimum_required(VERSION 3.10)
project(xtrim_yolo_detect)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ожидаем, что ncnn собран и установлен в android_native/ncnn-install
set(NCNN_INSTALL_DIR "${CMAKE_CURRENT_LIST_DIR}/ncnn-install")
set(NCNN_INCLUDE_DIR "${NCNN_INSTALL_DIR}/include")
set(NCNN_LIB_DIR "${NCNN_INSTALL_DIR}/lib")

include_directories(${NCNN_INCLUDE_DIR})
link_directories(${NCNN_LIB_DIR})

add_executable(xtrim_yolo_detect yolo_detect.cpp)

# ncnn обычно статическая libncnn.a + системные
target_link_libraries(xtrim_yolo_detect
    ncnn
    z
    log
)