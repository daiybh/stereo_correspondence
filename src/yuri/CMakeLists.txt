#################################################################
# Core library depends only on Boost 
#################################################################
find_package( Boost 1.46 COMPONENTS thread system filesystem REQUIRED)

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

#################################################################
# Set up dependencies for libyuri_core
#################################################################
SET (YURI_LINK ${YURI_LINK} ${Boost_LIBRARY_DIRS} )
SET (YURI_LIBS ${YURI_LIBS}	${Boost_LIBRARIES} )
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

IF(${CUDA_FOUND})
	SET(YURI_HELPER_CUDA_INCLUDE ${YURI_HELPER_CUDA_INCLUDE} ${CUDA_INCLUDE_DIRS})
	SET(YURI_HELPER_CUDA_LIBS ${YURI_HELPER_CUDA_LIBS} ${CUDA_LIBRARIES})
	SET(YURI_HELPER_CUDA_LINK ${YURI_HELPER_CUDA_LINK} ${CUDA_LIBRARY_DIRS})
ENDIF()

#################################################################
# Adding CXXFLAGS and LDFLAGS for libyuri_core
#################################################################
add_definitions("-DYURI_MODULE_IN_TREE -DINSTALL_PREFIX=\\\"${CMAKE_INSTALL_PREFIX}\\\"")
link_directories ( ${YURI_LINK} ${YURI_HELPER_LIBAV_LINK} ${YURI_HELPER_X11_LINK})
include_directories ( ${YURI_INCLUDE} ${YURI_HELPER_LIBAV_INCLUDE} ${YURI_HELPER_X11_INCLUDE})

#################################################################
# Submerge into subdirectories to collect source files
#################################################################
add_subdirectory(log)
add_subdirectory(config)
add_subdirectory(threads)
add_subdirectory(exception)
add_subdirectory(io)

#################################################################
# Collect sources for helper libraries
#################################################################
IF(${libav_FOUND})
	add_subdirectory(libav)
	SET(CORE_LIBAV TRUE PARENT_SCOPE)
	SET(CORE_LIBAV TRUE)
	add_definitions("-DYURI_HAVE_LIBAV")
ENDIF()
IF(${CUDA_FOUND})
	add_subdirectory(cuda)
	SET(CORE_CUDA TRUE PARENT_SCOPE)
	SET(CORE_CUDA TRUE)
	add_definitions("-DYURI_HAVE_CUDA")
ENDIF()
IF(${X11_FOUND})
	IF(${OPENGL_FOUND})
		add_subdirectory(graphics)
		SET(CORE_X11 TRUE PARENT_SCOPE)
		SET(CORE_X11 TRUE)
		add_definitions("-DYURI_HAVE_X11")
	ENDIF()
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
	SET_TARGET_PROPERTIES( ${LIBNAME_HELPER_CUDA} PROPERTIES SOVERSION 2 VERSION 2.5.0 )
	install(TARGETS ${LIBNAME_HELPER_CUDA} 	RUNTIME DESTINATION bin
						LIBRARY DESTINATION lib 
						ARCHIVE DESTINATION lib)
	
ENDIF()

IF(${CORE_LIBAV})
	MESSAGE("Building libav helper library!")
	add_library(${LIBNAME_HELPER_LIBAV} ${YURI_HELPER_LIBAV_SRC})
	target_link_libraries ( ${LIBNAME_HELPER_LIBAV} ${YURI_HELPER_LIBAV_LIBS})
	SET_TARGET_PROPERTIES( ${LIBNAME_HELPER_LIBAV} PROPERTIES SOVERSION 2 VERSION 2.5.0 )
	install(TARGETS ${LIBNAME_HELPER_LIBAV} 	RUNTIME DESTINATION bin
						LIBRARY DESTINATION lib 
						ARCHIVE DESTINATION lib)
	
ENDIF()

IF(${CORE_X11})
	MESSAGE("Building X11 helper library!")
	add_library(${LIBNAME_HELPER_X11} ${YURI_HELPER_X11_SRC})
	target_link_libraries ( ${LIBNAME_HELPER_X11} ${YURI_HELPER_X11_LIBS})
	SET_TARGET_PROPERTIES( ${LIBNAME_HELPER_X11} PROPERTIES SOVERSION 2 VERSION 2.5.0 )
	install(TARGETS ${LIBNAME_HELPER_X11} 	RUNTIME DESTINATION bin
						LIBRARY DESTINATION lib 
						ARCHIVE DESTINATION lib)
ENDIF()


IF(UNIX)
	SET( YURI_LIBS ${YURI_LIBS} rt dl)
ENDIF()

#################################################################
# Set linking for libyuri_core
#################################################################
target_link_libraries ( ${LIBNAME} ${YURI_LIBS})

SET_TARGET_PROPERTIES( ${LIBNAME} PROPERTIES SOVERSION 2 VERSION 2.5.0 )
install(TARGETS ${LIBNAME} 	RUNTIME DESTINATION bin
						LIBRARY DESTINATION lib 
						ARCHIVE DESTINATION lib)
install(DIRECTORY ../yuri DESTINATION include
          				FILES_MATCHING PATTERN "*.h")
               

						