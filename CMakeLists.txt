cmake_minimum_required(VERSION 2.8)  
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
project(kinectFusionT)

#OPENCV
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} )
link_directories(${OPENCV_LIBRARY_DIRS})
add_definitions(${OPENCV_DEFINITIONS})

add_executable(kinectFusionT main.cpp)

# Find Kinect SDK v2
#find_package(KinectSDK2 REQUIRED)
# Settings Kinect SDK v2

include_directories("C:/Program Files/Microsoft SDKs/Kinect/v2.0_1409/inc")
#link_directories("C:/Program Files/Microsoft SDKs/Kinect/v2.0_1409/Lib/x64")

  #add_custom_command( TARGET project POST_BUILD ${KinectSDK2_COMMANDS} )

TARGET_LINK_LIBRARIES(kinectFusionT ${OpenCV_LIBS})