cmake_minimum_required(VERSION 3.22)
project(yoloapp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_ANDROID_STL_TYPE c++_shared)

set(SRC_DIR ${CMAKE_SOURCE_DIR})
set(SOURCES)
foreach(f
        yolo_jni.cpp
        yolov8.cpp
        classifier_jni.cpp
        classifier.cpp
        fastcnn_jni.cpp
        fasterrcnn.cpp
)
    if(EXISTS ${SRC_DIR}/${f})
        list(APPEND SOURCES ${f})
        message(STATUS "Add source: ${f}")
    else()
        message(STATUS "Skip missing source: ${f}")
    endif()
endforeach()

if(NOT SOURCES)
    message(FATAL_ERROR "No native source files found in ${SRC_DIR}")
endif()

add_library(yolo SHARED ${SOURCES})

set(NCNN_SO "${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libncnn.so")
if(NOT EXISTS "${NCNN_SO}")
    message(FATAL_ERROR "libncnn.so not found at: ${NCNN_SO}")
endif()

set(NCNN_INCLUDE_DIR "")
if(EXISTS "${CMAKE_SOURCE_DIR}/ncnn/include/ncnn/net.h")
    set(NCNN_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/ncnn/include")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/ncnn/net.h")
    set(NCNN_INCLUDE_DIR "${CMAKE_SOURCE_DIR}")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/ncnn/ncnn/net.h")
    set(NCNN_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/ncnn")
endif()

if(NOT NCNN_INCLUDE_DIR)
    message(FATAL_ERROR "ncnn headers not found.")
endif()
message(STATUS "NCNN include dir: ${NCNN_INCLUDE_DIR}")

add_library(ncnn SHARED IMPORTED GLOBAL)
set_target_properties(ncnn PROPERTIES
        IMPORTED_LOCATION "${NCNN_SO}"
        INTERFACE_INCLUDE_DIRECTORIES "${NCNN_INCLUDE_DIR}"
)

target_include_directories(yolo PRIVATE "${NCNN_INCLUDE_DIR}")

find_library(log-lib         log)
find_library(android-lib     android)
find_library(jnigraphics-lib jnigraphics)
find_library(cpp_shared      c++_shared)

target_link_libraries(yolo
        ncnn
        ${cpp_shared}
        ${log-lib}
        ${android-lib}
        ${jnigraphics-lib}
)

if(CMAKE_ANDROID_ARCH_ABI MATCHES "arm64-v8a|armeabi-v7a")
    target_compile_definitions(yolo PRIVATE __ARM_NEON=1)
endif()
