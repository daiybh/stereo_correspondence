cmake_minimum_required (VERSION 2.8)
PROJECT(yuri CXX)
ENABLE_LANGUAGE(C)

INCLUDE(FindPkgConfig)
INCLUDE(CheckIncludeFiles)

IF (UNIX)
add_definitions("-Wall -pedantic -Wextra")
ENDIF ()
OPTION (BUILD_SHARED_LIBS "Build shared libraries." ON)
SET(LIBNAME yuri_light)

include_directories( ${CMAKE_SOURCE_DIR}/src )

add_subdirectory(src)



#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/libmvtp.pc.in ${CMAKE_CURRENT_BINARY_DIR}/libmvtp.pc @ONLY )
#install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libmvtp.pc DESTINATION lib/pkgconfig)

#add_subdirectory(include)
#add_subdirectory(examples)
