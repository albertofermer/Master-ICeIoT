cmake_minimum_required (VERSION 3.0)
project (ARBlockMuseum)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

LINK_LIBRARIES(${OpenCV_LIBS})

add_executable(armuseum armuseum.cpp rva.cpp)




