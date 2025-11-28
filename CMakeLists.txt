# CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(SVVRP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 调试时默认带 -g -O0，Release 时你再改成 -O2
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -w")

# 自动收集当前目录所有 .cpp
file(GLOB SOURCES "*.cpp")

set(DATA_FILES 
    "tsp数据坐标集.txt"      # ← 改成你的真实文件名
)

foreach(file ${DATA_FILES})
    if(EXISTS "${CMAKE_SOURCE_DIR}/${file}")
        configure_file("${CMAKE_SOURCE_DIR}/${file}" "${CMAKE_BINARY_DIR}/${file}" COPYONLY)
        message(STATUS "Copy data file: ${file}")
    else()
        message(WARNING "Data file not found: ${file}")
    endif()
endforeach()

# 生成可执行文件叫 app
add_executable(app ${SOURCES})