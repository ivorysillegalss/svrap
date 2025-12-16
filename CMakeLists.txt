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

# Copy the entire formatted_dataset directory to the build folder
if(EXISTS "${CMAKE_SOURCE_DIR}/formatted_dataset")
    file(COPY "${CMAKE_SOURCE_DIR}/formatted_dataset" DESTINATION "${CMAKE_BINARY_DIR}")
    message(STATUS "Copied formatted_dataset to build directory")
else()
    message(WARNING "formatted_dataset directory not found")
endif()

# 生成可执行文件叫 app
add_executable(app ${SOURCES})