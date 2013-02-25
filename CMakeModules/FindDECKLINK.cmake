# - Find the decklink API includes and library
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
#  To override automatic detection, please specify environmental variables:
#  DECKLINK_INCLUDE  	pointing to directory with decklink includes
#  DECKLINK_LIBDIR 		pointing to directory with decklink libraries


SET(DECKLINK_INC $ENV{DECKLINK_INCLUDE})
SET(DECKLINK_LIBDIR $ENV{DECKLINK_LIBDIR})

find_path(DECKLINK_DECKLINK_INCLUDE_DIR DeckLinkAPI.h PATHS
  ${DECKLINK_INC}
  /usr/local/include/decklink
  /usr/local/include/decklink/include
  /usr/include/decklink
  /usr/include/decklink/include
  /usr/include
  /usr/local/include    
 )

  
find_library(DECKLINK_DECKLINK_LIBRARY DeckLinkAPI PATH 
	${DECKLINK_LIBDIR}
 	/usr/lib/
 	/usr/local/lib)
  
IF (DECKLINK_DECKLINK_LIBRARY AND DECKLINK_DECKLINK_INCLUDE_DIR)
	SET(DECKLINK_INCLUDE_DIRS ${DECKLINK_DECKLINK_INCLUDE_DIR} )
	SET(DECKLINK_INCLUDE_DIR ${DECKLINK_DECKLINK_INCLUDE_DIR} ) # for backward compatiblity
	SET(DECKLINK_LIBRARY ${DECKLINK_DECKLINK_LIBRARY})
	SET(DECKLINK_LIBRARIES ${DECKLINK_DECKLINK_LIBRARY})
	if (DECKLINK_DECKLINK_INCLUDE_DIR AND EXISTS "${DECKLINK_DECKLINK_INCLUDE_DIR}/DeckLinkAPIVersion.h")
		file(STRINGS "${DECKLINK_DECKLINK_INCLUDE_DIR}/DeckLinkAPIVersion.h" decklink_version_str REGEX "^#define[ \t]+BLACKMAGIC_DECKLINK_API_VERSION_STRING[ \t]+\".+\"")
		string(REGEX REPLACE "^#define[ \t]+BLACKMAGIC_DECKLINK_API_VERSION_STRING[ \t]+\"([^\"]+)\".*" "\\1" DECKLINK_VERSION_STRING "${decklink_version_str}")
		unset(decklink_version_str)
		#MESSAGE(${DECKLINK_VERSION_STRING})
	endif ()
ENDIF()      

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DECKLINK  
                                  REQUIRED_VARS DECKLINK_LIBRARY DECKLINK_INCLUDE_DIR
                                  VERSION_VAR DECKLINK_VERSION_STRING)


mark_as_advanced(DECKLINK_DECKLINK_INCLUDE_DIR DECKLINK_DECKLINK_LIBRARY )
