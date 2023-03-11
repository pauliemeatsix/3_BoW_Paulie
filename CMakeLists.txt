cmake_minimum_required(VERSION 3.1)
project(project_bovw)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC")

find_package(OpenCV 4.6 REQUIRED)
if(OpenCV_FOUND)
  message(STATUS "Found OpenCV version ${OpenCV_VERSION}")
  message(STATUS "OpenCV directories: ${OpenCV_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "OpenCV not found, please read the README.md")
endif(OpenCV_FOUND)
find_package(TBB REQUIRED tbb)

include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories("include")

add_subdirectory(lib)

add_executable(main src/main.cpp)
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main tbb)

target_link_libraries(main image_browser)
target_link_libraries(main html_writer)
target_link_libraries(main convert_dataset)
target_link_libraries(main serialize)
target_link_libraries(main bow)
target_link_libraries(main kmeans)


#for the main.hpp later 
# target_include_directories(project INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

