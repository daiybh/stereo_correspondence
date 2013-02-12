# - Find the sagelib includes and library
#
# This module searches libsage, the library for working with PNG images.
#
# It defines the following variables
#  PNG_INCLUDE_DIRS, where to find png.h, etc.
#  PNG_LIBRARIES, the libraries to link against to use PNG.
#  PNG_DEFINITIONS - You should add_definitons(${PNG_DEFINITIONS}) before compiling code that includes png library files.
#  PNG_FOUND, If false, do not try to use PNG.
#  PNG_VERSION_STRING - the version of the PNG library found (since CMake 2.8.8)
# Also defined, but not for general use are
#  PNG_LIBRARY, where to find the PNG library.
# For backward compatiblity the variable PNG_INCLUDE_DIR is also set. It has the same value as PNG_INCLUDE_DIRS.
#
# Since PNG depends on the ZLib compression library, none of the above will be
# defined unless ZLib can be found.

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

find_path(SAGE_SAGE_INCLUDE_DIR libsage.h
  /usr/local/sage/include             # Default installation folder
 )

  
  find_library(SAGE_SAIL_LIBRARY NAMES sail )
  

  if (SAGE_SAIL_LIBRARY AND SAGE_SAGE_INCLUDE_DIR)
      # png.h includes zlib.h. Sigh.
      SET(SAGE_INCLUDE_DIRS ${SAGE_SAGE_INCLUDE_DIR} )
      SET(SAGE_INCLUDE_DIR ${SAGE_SAGE_INCLUDE_DIRS} ) # for backward compatiblity
      SET(SAGE_LIBRARY ${SAGE_SAIL_LIBRARY})
      SET(SAGE_VERSION "3.0")
	endif()      
# handle the QUIETLY and REQUIRED arguments and set PNG_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SAGE
                                  REQUIRED_VARS SAGE_LIBRARY SAGE_SAGE_INCLUDE_DIR
                                  VERSION_VAR SAGE_VERSION)

mark_as_advanced(SAGE_SAGE_INCLUDE_DIR PNG_LIBRARY )
