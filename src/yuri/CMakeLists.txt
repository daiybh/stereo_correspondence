#################################################################
# Core library can use boost filessytem 
#################################################################
IF(NOT YURI_DISABLE_BOOST)
	find_package( Boost 1.49 COMPONENTS filesystem system)
	IF(Boost_FILESYSTEM_FOUND)
		add_definitions(-DHAVE_BOOST_FILESYSTEM)
	ENDIF()
ENDIF()
#################################################################
# Checking for optional dependencies for helper libraries
#################################################################

IF(NOT YURI_DISABLE_X11)
	find_package( X11 )	
	find_package( OpenGL )
ENDIF()
IF(NOT YURI_DISABLE_LIBAV)
	pkg_check_modules(libav libavcodec)
ENDIF()
IF(NOT YURI_DISABLE_CUDA)
	find_package( CUDA)
ENDIF()

CHECK_INCLUDE_FILES (stdint.h HAVE_STDINT_H)

IF (NOT HAVE_STDINT_H)
	MESSAGE(FATAL_ERROR "Missing stdint.h. Please update your compile chain")
ENDIF()

IF(NOT YURI_DISABLE_IEEE1394)
	pkg_check_modules(ieee1394 libiec61883 libavc1394)
ENDIF()

# This wont't work as the linking for this program fails with unresolved symbols from libboost_system
#IF(NOT YURI_DISABLE_ASIO)
#	CHECK_INCLUDE_FILE_CXX ("boost/asio.hpp" HAVE_BOOST_ASIO_H)
#ENDIF()

IF(NOT YURI_DISABLE_DECKLINK)
	find_package(DECKLINK)
ENDIF()

IF(NOT YURI_DISABLE_ULTRAGRID)
	# This should be replaced by a proper detection!
	SET(UV_BASE ${CMAKE_SOURCE_DIR}/../ultragrid/src)
	SET(ULTRAGRID_FOUND TRUE)
	find_package(X11)
	find_package(OpenSSL)
	pkg_check_modules(SPEEX speex speexdsp)
ENDIF()

#################################################################
# Set up dependencies for libyuri_core
#################################################################
SET (YURI_LINK ${YURI_LINK} ${Boost_LIBRARY_DIRS} )
IF(NOT WIN32)
SET (YURI_LIBS ${YURI_LIBS}	${Boost_LIBRARIES} )
ENDIF()
SET (YURI_INCLUDE ${YURI_INCLUDE} ${Boost_INCLUDE_DIRS} )  

#################################################################
# Set up dependencies for helper libraries
#################################################################
SET (YURI_HELPER_LIBAV_LINK ${YURI_HELPER_LINK} ${libav_LIBRARY_DIRS})
SET (YURI_HELPER_X11_LINK ${YURI_HELPER_LINK} ${X11_LIBRARY_DIRS})
		
SET (YURI_HELPER_LIBAV_LIBS ${YURI_HELPER_LIBAV_LIBS} 
		${LIBNAME} ${libav_LIBRARIES})
SET (YURI_HELPER_X11_LIBS ${YURI_HELPER_X11_LIBS} 
		${LIBNAME} ${X11_LIBRARIES}	${OPENGL_gl_LIBRARY})

SET (YURI_HELPER_LIBAV_INCLUDE  ${YURI_HELPER_LIBAV_INCLUDE}  ${libav_INCLUDE_DIRS})
SET (YURI_HELPER_X11_INCLUDE ${YURI_HELPER_X11_INCLUDE} ${X11_INCLUDE_DIRS} )

IF(NOT YURI_DISABLE_CUDA AND CUDA_FOUND)
	SET(YURI_HELPER_CUDA_INCLUDE ${YURI_HELPER_CUDA_INCLUDE} ${CUDA_INCLUDE_DIRS})
	SET(YURI_HELPER_CUDA_LIBS ${LIBNAME} ${YURI_HELPER_CUDA_LIBS} ${CUDA_LIBRARIES})
	SET(YURI_HELPER_CUDA_LINK ${YURI_HELPER_CUDA_LINK} ${CUDA_LIBRARY_DIRS})
ENDIF()

IF(NOT YURI_DISABLE_IEEE1394 AND ieee1394_FOUND)
	SET(YURI_HELPER_IEEE1394_INCLUDE  ${ieee1394_INCLUDE_DIRS})
	SET(YURI_HELPER_IEEE1394_LIBS ${LIBNAME} ${ieee1394_LIBRARIES})
	SET(YURI_HELPER_IEEE1394_LINK  ${ieee1394_LIBRARY_DIRS})
ENDIF()

IF(NOT YURI_DISABLE_ASIO)
	SET(YURI_HELPER_ASIO_INCLUDE  ${Boost_INCLUDE_DIRS})
ENDIF()

IF(NOT YURI_DISABLE_DECKLINK AND DECKLINK_FOUND)
	SET(YURI_HELPER_DECKLINK_INCLUDE ${DECKLINK_INCLUDE_DIRS})
	SET(YURI_HELPER_DECKLINK_LIBS ${DECKLINK_LIBRARIES})
ENDIF()

IF(NOT YURI_DISABLE_ULTRAGRID)
	SET(YURI_HELPER_UV_INCLUDE ${UV_BASE})
	SET(YURI_HELPER_UV_LIBS ${LIBNAME} ${X11_LIBRARIES} ${X11_Xfixes_LIB} ${OPENSSL_LIBRARIES} ${SPEEX_LIBRARIES} ${SPEEXDSP_LIBRARIES})
ENDIF()
#################################################################
# Adding CXXFLAGS and LDFLAGS for libyuri_core
#################################################################
add_definitions("-DYURI_MODULE_IN_TREE -DINSTALL_PREFIX=\\\"${CMAKE_INSTALL_PREFIX}\\\"")
link_directories ( ${YURI_LINK} ${YURI_HELPER_LIBAV_LINK} ${YURI_HELPER_X11_LINK})
include_directories ( ${YURI_INCLUDE} ${YURI_HELPER_LIBAV_INCLUDE} ${YURI_HELPER_X11_INCLUDE} ${YURI_HELPER_DECKLINK_INCLUDE})

#################################################################
# Submerge into subdirectories to collect source files
#################################################################
add_subdirectory(log)
add_subdirectory(exception)
add_subdirectory(core)
add_subdirectory(event)
#################################################################
# Collect sources for helper libraries
#################################################################
IF(NOT YURI_DISABLE_LIBAV AND libav_FOUND)
	add_subdirectory(libav)
	SET(CORE_LIBAV TRUE PARENT_SCOPE)
	SET(CORE_LIBAV TRUE)
	add_definitions("-DYURI_HAVE_LIBAV")
ELSE()
	SET(CORE_LIBAV FALSE PARENT_SCOPE)
	SET(CORE_LIBAV FALSE)
ENDIF()
IF(NOT YURI_DISABLE_CUDA AND CUDA_FOUND)
	add_subdirectory(cuda)
	SET(CORE_CUDA TRUE PARENT_SCOPE)
	SET(CORE_CUDA TRUE)
	add_definitions("-DYURI_HAVE_CUDA")
ELSE()
	SET(CORE_CUDA FALSE PARENT_SCOPE)
	SET(CORE_CUDA FALSE)
ENDIF()
IF(NOT YURI_DISABLE_X11 AND X11_FOUND AND OPENGL_FOUND)
	add_subdirectory(graphics)
	SET(CORE_X11 TRUE PARENT_SCOPE)
	SET(CORE_X11 TRUE)
	add_definitions("-DYURI_HAVE_X11")
ELSE()
	SET(CORE_X11 FALSE PARENT_SCOPE)
	SET(CORE_X11 FALSE)
ENDIF()
IF(NOT YURI_DISABLE_IEEE1394 AND ieee1394_FOUND)
	add_subdirectory(ieee1394)
	SET(CORE_IEEE1394 TRUE PARENT_SCOPE)
	SET(CORE_IEEE1394 TRUE)
	add_definitions("-DYURI_HAVE_IEEE1394")
ELSE()
	SET(CORE_IEEE1394 FALSE PARENT_SCOPE)
	SET(CORE_IEEE1394 FALSE)
ENDIF()
IF(NOT YURI_DISABLE_ASIO)
	add_subdirectory(asio)
	SET(CORE_ASIO TRUE PARENT_SCOPE)
	SET(CORE_ASIO TRUE)
	add_definitions("-DYURI_HAVE_ASIO")
ELSE()
	SET(CORE_ASIO FALSE PARENT_SCOPE)
	SET(CORE_ASIO FALSE)
ENDIF()

IF(NOT YURI_DISABLE_DECKLINK AND DECKLINK_FOUND)
	add_subdirectory(decklink)
	SET(CORE_DECKLINK TRUE PARENT_SCOPE)
	SET(CORE_DECKLINK TRUE)
	add_definitions("-DYURI_HAVE_DECKLINK")
ELSE()
	SET(CORE_DECKLINK FALSE PARENT_SCOPE)
	SET(CORE_DECKLINK FALSE)
ENDIF()

IF(NOT YURI_DISABLE_ULTRAGRID AND ULTRAGRID_FOUND)
	add_subdirectory(ultragrid)
	SET(CORE_UV TRUE PARENT_SCOPE)
	SET(CORE_UV TRUE)
	add_definitions("-DYURI_HAVE_UV")
ELSE()
	SET(CORE_UV FALSE PARENT_SCOPE)
	SET(CORE_UV FALSE)
ENDIF()

#################################################################
# Generate version.cpp with current git revision
#################################################################
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/version.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/version.cpp" @ONLY)
SET(YURI_SRC ${YURI_SRC} "${CMAKE_CURRENT_BINARY_DIR}/version.cpp" ${CMAKE_CURRENT_SOURCE_DIR}/version.h )

#################################################################
# Add target for libyuri_core
#################################################################
add_library(${LIBNAME} ${YURI_SRC})

#################################################################
# Add targets for helper libraries
#################################################################
IF(${CORE_CUDA})
	MESSAGE("Building CUDA helper library!")
	IF(${CORE_X11})
		SET(YURI_HELPER_CUDA_LIBS ${YURI_HELPER_CUDA_LIBS} ${LIBNAME_HELPER_X11})
	ENDIF()
	cuda_add_library(${LIBNAME_HELPER_CUDA} ${YURI_HELPER_CUDA_SRC})
	target_link_libraries ( ${LIBNAME_HELPER_CUDA} ${YURI_HELPER_CUDA_LIBS})
	SET_TARGET_PROPERTIES( ${LIBNAME_HELPER_CUDA} PROPERTIES SOVERSION ${SO_VERSION} VERSION ${YURI_CORE_VERSION} )
	install(TARGETS ${LIBNAME_HELPER_CUDA} 	RUNTIME DESTINATION bin
						LIBRARY DESTINATION lib 
						ARCHIVE DESTINATION lib)
	
ENDIF()

IF(${CORE_LIBAV})
	MESSAGE("Building libav helper library!")
	add_library(${LIBNAME_HELPER_LIBAV} ${YURI_HELPER_LIBAV_SRC})
	target_link_libraries ( ${LIBNAME_HELPER_LIBAV} ${YURI_HELPER_LIBAV_LIBS})
	SET_TARGET_PROPERTIES( ${LIBNAME_HELPER_LIBAV} PROPERTIES SOVERSION ${SO_VERSION} VERSION ${YURI_CORE_VERSION} )
	install(TARGETS ${LIBNAME_HELPER_LIBAV} 	RUNTIME DESTINATION bin
						LIBRARY DESTINATION lib 
						ARCHIVE DESTINATION lib)
	
ENDIF()

IF(${CORE_X11})
	MESSAGE("Building X11 helper library!")
	add_library(${LIBNAME_HELPER_X11} ${YURI_HELPER_X11_SRC})
	target_link_libraries ( ${LIBNAME_HELPER_X11} ${YURI_HELPER_X11_LIBS})
	SET_TARGET_PROPERTIES( ${LIBNAME_HELPER_X11} PROPERTIES SOVERSION ${SO_VERSION} VERSION ${YURI_CORE_VERSION} )
	install(TARGETS ${LIBNAME_HELPER_X11} 	RUNTIME DESTINATION bin
						LIBRARY DESTINATION lib 
						ARCHIVE DESTINATION lib)
ENDIF()

IF(${CORE_IEEE1394})
	MESSAGE("Building ieee1394 helper library!")
	add_library(${LIBNAME_HELPER_IEEE1394} ${YURI_HELPER_IEEE1394_SRC})
	target_link_libraries ( ${LIBNAME_HELPER_IEEE1394} ${YURI_HELPER_IEEE1394_LIBS})
	SET_TARGET_PROPERTIES( ${LIBNAME_HELPER_IEEE1394} PROPERTIES SOVERSION ${SO_VERSION} VERSION ${YURI_CORE_VERSION} )
	install(TARGETS ${LIBNAME_HELPER_IEEE1394} 	RUNTIME DESTINATION bin
						LIBRARY DESTINATION lib 
						ARCHIVE DESTINATION lib)
ENDIF()
IF(${CORE_ASIO})
	MESSAGE("Building ASIO helper library!")
	add_library(${LIBNAME_HELPER_ASIO} ${YURI_HELPER_ASIO_SRC})
	target_link_libraries ( ${LIBNAME_HELPER_ASIO} ${YURI_HELPER_ASIO_LIBS})
	SET_TARGET_PROPERTIES( ${LIBNAME_HELPER_ASIO} PROPERTIES SOVERSION ${SO_VERSION} VERSION ${YURI_CORE_VERSION} )
	install(TARGETS ${LIBNAME_HELPER_ASIO} 	RUNTIME DESTINATION bin
						LIBRARY DESTINATION lib 
						ARCHIVE DESTINATION lib)
ENDIF()
IF(${CORE_DECKLINK})
	MESSAGE("Building DECKLINK helper library!")
	add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/decklink/DeckLinkAPIDispatch.cpp"
	    	COMMAND ${CMAKE_COMMAND} -E copy_if_different
	    	"${DECKLINK_INCLUDE_DIR}/DeckLinkAPIDispatch.cpp"
	    	"${CMAKE_CURRENT_BINARY_DIR}/decklink/DeckLinkAPIDispatch.cpp"
	    )	  
	SET(YURI_HELPER_DECKLINK_SRC ${YURI_HELPER_DECKLINK_SRC} "${CMAKE_CURRENT_BINARY_DIR}/decklink/DeckLinkAPIDispatch.cpp")
	add_library(${LIBNAME_HELPER_DECKLINK} ${YURI_HELPER_DECKLINK_SRC})
	target_link_libraries ( ${LIBNAME_HELPER_DECKLINK} ${YURI_HELPER_DECKLINK_LIBS})
	SET_TARGET_PROPERTIES( ${LIBNAME_HELPER_DECKLINK} PROPERTIES SOVERSION ${SO_VERSION} VERSION ${YURI_CORE_VERSION} )
	install(TARGETS ${LIBNAME_HELPER_DECKLINK} 	RUNTIME DESTINATION bin
						LIBRARY DESTINATION lib 
						ARCHIVE DESTINATION lib)
	
ENDIF()
IF(${CORE_UV})
		
	MESSAGE("Building ULTRAGRID helper library!")
	add_library(${LIBNAME_HELPER_UV} ${YURI_HELPER_UV_SRC})
	target_link_libraries ( ${LIBNAME_HELPER_UV} ${YURI_HELPER_UV_LIBS})
	SET_TARGET_PROPERTIES( ${LIBNAME_HELPER_UV} PROPERTIES SOVERSION ${SO_VERSION} VERSION ${YURI_CORE_VERSION} )
	SET_TARGET_PROPERTIES(${LIBNAME_HELPER_UV} PROPERTIES COMPILE_FLAGS "-DHAVE_CONFIG_H -DUV_IN_YURI")
	SET_PROPERTY(TARGET ${LIBNAME_HELPER_UV} PROPERTY INCLUDE_DIRECTORIES ${YURI_HELPER_UV_INCLUDE}/..;${YURI_HELPER_UV_INCLUDE})
	#SET_TARGET_PROPERTIES(${LIBNAME_HELPER_UV} PROPERTIES INCLUDE_DIRECTORIES ${YURI_HELPER_UV_INCLUDE})
	
	
	

	include_directories( ${CMAKE_SOURCE_DIR}/src )
	install(TARGETS ${LIBNAME_HELPER_UV} 	RUNTIME DESTINATION bin
						LIBRARY DESTINATION lib 
						ARCHIVE DESTINATION lib)
ENDIF()
IF(UNIX)
	IF(APPLE)
		SET( YURI_LIBS ${YURI_LIBS} dl)
	ELSE()
		SET( YURI_LIBS ${YURI_LIBS} rt dl)
	ENDIF()
ENDIF()
#################################################################
# Set linking for libyuri_core
#################################################################
target_link_libraries ( ${LIBNAME} ${YURI_LIBS})

SET_TARGET_PROPERTIES( ${LIBNAME} PROPERTIES SOVERSION ${SO_VERSION} VERSION ${YURI_CORE_VERSION} )
install(TARGETS ${LIBNAME} 	RUNTIME DESTINATION bin
						LIBRARY DESTINATION lib 
						ARCHIVE DESTINATION lib)
install(DIRECTORY ../yuri DESTINATION include
          				FILES_MATCHING PATTERN "*.h")
               

						
