cmake_minimum_required (VERSION 2.8)
PROJECT(yuri2 CXX)
ENABLE_LANGUAGE(C)

#################################################################
# Include required CMake modules
#################################################################
INCLUDE(FindPkgConfig)
INCLUDE(CheckIncludeFiles)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/")
include(cuda_module)
include(GetGitRevisionDescription)

#################################################################
# Get the version of this build
#################################################################
git_describe(YURI_GIT_VERSION)

#################################################################
# Add strict warning if building under linux
#################################################################
IF (UNIX)
add_definitions("-Wall -pedantic -Wextra")
ENDIF ()

#################################################################
# Setup names for the core library and helpers
#################################################################
SET(LIBNAME yuri_core)
SET(LIBNAME_HELPER_LIBAV yuri_helper_av)
SET(LIBNAME_HELPER_CUDA yuri_helper_cuda)
SET(LIBNAME_HELPER_X11 yuri_helper_x11)

#################################################################
# Set prefix for modules
#################################################################
SET (CMAKE_SHARED_MODULE_PREFIX "yuri2_module_")

#################################################################
# Setup paths
#################################################################
include_directories( ${CMAKE_SOURCE_DIR}/src )
SET (EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
SET (MODULE_DIR ${EXECUTABLE_OUTPUT_PATH}/modules)
SET (TARGET_MODULE_PATH lib/yuri2/)

#################################################################
# Initially disable all helpers
#################################################################
SET(CORE_LIBAV FALSE)
SET(CORE_CUDA FALSE)
SET(CORE_X11 FALSE)

#################################################################
# Add some options to control build
#################################################################
OPTION (BUILD_SHARED_LIBS "Build shared libraries." ON)
OPTION (YURI_DISABLE_CUDA "Disable building of CUDA helpers" OFF)
OPTION (YURI_DISABLE_LIBAV "Disable building of libav helpers and support" OFF)
OPTION (YURI_DISABLE_SAGE "Disable building of libsage related modules" OFF)
OPTION (YURI_DISABLE_X11 "Explicitly disable building of X11 helpers and related modules" OFF)
OPTION (YURI_DISABLE_PNG "Disable building of CUDA helpers" OFF)

#################################################################
# And finally submerge into source directories
#################################################################
add_subdirectory(src)

