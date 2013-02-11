cmake_minimum_required (VERSION 2.8)
PROJECT(yuri2 CXX)
ENABLE_LANGUAGE(C)

INCLUDE(FindPkgConfig)
INCLUDE(CheckIncludeFiles)

IF (UNIX)
add_definitions("-Wall -pedantic -Wextra")
ENDIF ()
OPTION (BUILD_SHARED_LIBS "Build shared libraries." ON)
SET(LIBNAME yuri_light)
SET (CMAKE_SHARED_MODULE_PREFIX "yuri2_module_")
include_directories( ${CMAKE_SOURCE_DIR}/src )
SET (EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
SET (MODULE_DIR ${EXECUTABLE_OUTPUT_PATH}/modules)
SET (TARGET_MODULE_PATH lib/yuri2/)
add_subdirectory(src)

