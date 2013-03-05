# - Find the sagelib includes and library
#
# This module searches libsage, the library for working with PNG images.
#
# It defines the following variables
#  SAGE_INCLUDE_DIRS, where to find png.h, etc.
#  SAGE_LIBRARIES, the libraries to link against to use PNG.
#  SAGE_DEFINITIONS - You should add_definitons(${PNG_DEFINITIONS}) before compiling code that includes png library files.
#  SAGE_FOUND, If false, do not try to use PNG.
#=============================================================================
# Copyright 2013 Zdenek Travnicek, Institute of Intermedia (www.iim.cz)
#
# based on FindPNG.cmake from cmake distribution 
# Copyright 2002-2009 Kitware, Inc.
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)

SET(SAGE_INSTALL_DIR $ENV{SAGE_DIRECTORY})
IF("${SAGE_INSTALL_DIR}" STREQUAL "")
	# No SAGE_DIRECTORY set, so let's try defalt installation path
	SET(SAGE_INSTALL_DIR "/usr/local/sage") 
ENDIF()
find_path(SAGE_SAGE_INCLUDE_DIR libsage.h
  ${SAGE_INSTALL_DIR}/include
 )

  
  find_library(SAGE_SAIL_LIBRARY sail PATH ${SAGE_INSTALL_DIR}/lib64)
  

  if (SAGE_SAIL_LIBRARY AND SAGE_SAGE_INCLUDE_DIR)
      # png.h includes zlib.h. Sigh.
      SET(SAGE_INCLUDE_DIRS ${SAGE_SAGE_INCLUDE_DIR} )
      SET(SAGE_INCLUDE_DIR ${SAGE_SAGE_INCLUDE_DIRS} ) # for backward compatiblity
      SET(SAGE_LIBRARY ${SAGE_SAIL_LIBRARY})
      SET(SAGE_LIBRARIES ${SAGE_SAIL_LIBRARY})
      SET(SAGE_VERSION "3.0")
	endif()      

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SAGE
                                  REQUIRED_VARS SAGE_LIBRARY SAGE_SAGE_INCLUDE_DIR
                                  VERSION_VAR SAGE_VERSION)

mark_as_advanced(SAGE_SAGE_INCLUDE_DIR SAGE_SAGE_LIBRARY )
