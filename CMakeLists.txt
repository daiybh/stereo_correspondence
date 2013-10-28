cmake_minimum_required (VERSION 2.8.0)
PROJECT(yuri2 CXX)
ENABLE_LANGUAGE(C)

SET(OLD_CMAKE FALSE)
IF (CMAKE_MAJOR_VERSION EQUAL 2 AND CMAKE_MINOR_VERSION EQUAL 8 AND CMAKE_PATCH_VERSION LESS 4)
	SET(OLD_CMAKE TRUE)
	MESSAGE("Old cmake version detected, disabling generator expressions")
ENDIF()
#################################################################
# Include required CMake modules
#################################################################
INCLUDE(FindPkgConfig)
INCLUDE(CheckIncludeFiles)
INCLUDE(CheckIncludeFileCXX)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/")
include(cuda_module)
include(GetGitRevisionDescription)
include(yuri_install)
#################################################################
# Get the version of this build
#################################################################
git_describe(YURI_GIT_VERSION)

#################################################################
# Add strict warning if building under linux
#################################################################
IF (UNIX)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra -std=c++0x")
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -pedantic -Wextra -std=gnu11")

ENDIF ()

add_definitions("-DYURI_USE_CXX11") 
#################################################################
# Set versions
#################################################################
SET(YURI_CORE_VERSION 2.8)
SET(SO_VERSION 2.8.0)



#################################################################
# Setup names for the core library and helpers
#################################################################
SET(LIBNAME_PREFIX 			yuri${YURI_CORE_VERSION})
SET(LIBNAME 				${LIBNAME_PREFIX}_core)
SET(LIBNAME_HELPER_LIBAV 	${LIBNAME_PREFIX}_helper_av)
SET(LIBNAME_HELPER_CUDA 	${LIBNAME_PREFIX}_helper_cuda)
SET(LIBNAME_HELPER_X11 		${LIBNAME_PREFIX}_helper_x11)
SET(LIBNAME_HELPER_IEEE1394 ${LIBNAME_PREFIX}_helper_ieee1394)
SET(LIBNAME_HELPER_ASIO 	${LIBNAME_PREFIX}_helper_asio)
SET(LIBNAME_HELPER_DECKLINK ${LIBNAME_PREFIX}_helper_decklink)

SET(LIBNAME_HELPER_UV 		${LIBNAME_PREFIX}_helper_uv)

#################################################################
# Set prefix for modules
#################################################################
SET (CMAKE_SHARED_MODULE_PREFIX "yuri2.8_module_")

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
SET(CORE_IEEE1394 FALSE)
SET(CORE_ASIO FALSE)
SET(CORE_UV FALSE)

#################################################################
# Add some options to control build
#################################################################
OPTION (BUILD_SHARED_LIBS "Build shared libraries." ON)
OPTION (YURI_DISABLE_CUDA "Disable building of CUDA helpers" ON)
OPTION (YURI_DISABLE_LIBAV "Disable building of libav helpers and support" ON)
OPTION (YURI_DISABLE_SAGE "Disable building of libsage related modules" ON)
OPTION (YURI_DISABLE_X11 "Explicitly disable building of X11 helpers and related modules" ON)
OPTION (YURI_DISABLE_IEEE1394 "Disable building of IEEE1394 helpers" ON)
OPTION (YURI_DISABLE_ASIO "Disable building of boost::asio helpers" ON)
OPTION (YURI_DISABLE_DECKLINK "Disable building of decklink API helpers" ON)
OPTION (YURI_DISABLE_PNG "Disable building of CUDA helpers" ON)
OPTION (YURI_DISABLE_BOOST "Disable boost dependencies" OFF)

OPTION (YURI_DISABLE_ULTRAGRID "Disable ultragrid helper and modules" ON)

#################################################################
# And finally submerge into source directories
#################################################################
add_subdirectory(src)
add_subdirectory(doc)

