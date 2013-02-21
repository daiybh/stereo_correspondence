# - Find the sagelib includes and library
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

SET(OPENNI2_INC $ENV{OPENNI22_INCLUDE})
SET(OPENNI2_RED $ENV{OPENNI22_REDIST})
find_path(OPENNI2_OPENNI2_INCLUDE_DIR OpenNI.h PATHS
  ${OPENNI2_INC}
  /usr/local/include/OPENNI2  
  /usr/local/include/ni
  /usr/include/OPENNI2
  /usr/include/ni
 )

  
find_library(OPENNI2_OPENNI2_LIBRARY OpenNI2 PATH 
	${OPENNI2_REDIST}
 	/usr/lib/
 	/usr/local/lib)
  
  if (OPENNI2_OPENNI2_LIBRARY AND OPENNI2_OPENNI2_INCLUDE_DIR)
      SET(OPENNI2_INCLUDE_DIRS ${OPENNI2_OPENNI2_INCLUDE_DIR} )
      SET(OPENNI2_INCLUDE_DIR ${OPENNI2_OPENNI2_INCLUDE_DIR} ) # for backward compatiblity
      SET(OPENNI2_LIBRARY ${OPENNI2_OPENNI2_LIBRARY})
      SET(OPENNI2_LIBRARIES ${OPENNI2_OPENNI2_LIBRARY})
      SET(OPENNI2_VERSION "2.0")
	endif()      

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OPENNI2 DEFAULT_MSG 
                                  OPENNI2_LIBRARY OPENNI2_INCLUDE_DIR)
                                  #VERSION_VAR OPENNI2_VERSION)


mark_as_advanced(OPENNI2_OPENNI2_INCLUDE_DIR OPENNI2_OPENNI2_LIBRARY )
