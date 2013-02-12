SET (YURI_LINK )
SET (YURI_LIBS )
SET (YURI_INCLUDE )
SET (YURI_SRC )

find_package( Boost 1.46 COMPONENTS thread system filesystem REQUIRED)
find_package( X11  REQUIRED)
find_package( OpenGL REQUIRED)
pkg_check_modules(libav libavcodec)
find_package( CUDA)

CHECK_INCLUDE_FILES (stdint.h HAVE_STDINT_H)

IF (NOT HAVE_STDINT_H)
	MESSAGE(FATAL_ERROR "Missing stdint.h. Please update your compile chain")
ENDIF()


SET (YURI_LINK ${YURI_LINK} 
		${Boost_LIBRARY_DIRS} 
		${X11_LIBRARY_DIRS}
		${libav_LIBRARY_DIRS})
		
SET (YURI_LIBS ${YURI_LIBS} 
		${Boost_LIBRARIES}  
		${X11_LIBRARIES}
		${OPENGL_gl_LIBRARY}
		${libav_LIBRARIES})
SET (YURI_INCLUDE ${YURI_INCLUDE} 
		${Boost_INCLUDE_DIRS}  
		${X11_INCLUDE_DIRS}
		${libav_INCLUDE_DIRS}
		)
SET (YURI_SRC ${YURI_SRC} )
IF(${CUDA_FOUND})
	SET(YURI_INCLUDE ${YURI_INCLUDE} ${CUDA_INCLUDE_DIRS})
	SET(YURI_LIBS ${YURI_LIBS} ${CUDA_LIBRARIES})
	SET(YURI_LINK ${YURI_LINK} ${CUDA_LIBRARY_DIRS})
ENDIF()

add_definitions("-DYURI_MODULE_IN_TREE -DINSTALL_PREFIX=\\\"${CMAKE_INSTALL_PREFIX}\\\"")
link_directories ( ${YURI_LINK} )
include_directories ( ${YURI_INCLUDE} )



add_subdirectory(log)
add_subdirectory(config)
add_subdirectory(threads)
add_subdirectory(graphics)
add_subdirectory(exception)
add_subdirectory(io)

IF(${libav_FOUND})
	add_subdirectory(libav)
	SET(CORE_LIBAV TRUE PARENT_SCOPE)
ENDIF()
IF(${CUDA_FOUND})
	add_subdirectory(cuda)
	SET(CORE_CUDA TRUE PARENT_SCOPE)
	SET(CORE_CUDA TRUE)
ENDIF()

#SET (YURI_HEADERS )
#FOREACH(item ${YURI_SRC})
#std::string(REGEX MATCH ".*[.]h$" tmp ${item}) 
#	SET(YURI_HEADERS ${YURI_HEADERS} ${tmp})
#ENDFOREACH(item)


configure_file("${CMAKE_CURRENT_SOURCE_DIR}/version.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/version.cpp" @ONLY)
SET(YURI_SRC ${YURI_SRC} "${CMAKE_CURRENT_BINARY_DIR}/version.cpp" ${CMAKE_CURRENT_SOURCE_DIR}/version.h )

IF(${CORE_CUDA})
	MESSAGE("Building CUDA library!")
	cuda_add_library(${LIBNAME} ${YURI_SRC})
ELSE()
	add_library(${LIBNAME} ${YURI_SRC})
ENDIF()




IF(UNIX)
target_link_libraries ( ${LIBNAME} ${YURI_LIBS} rt)
ENDIF()
SET_TARGET_PROPERTIES( ${LIBNAME} PROPERTIES SOVERSION 2 VERSION 2.5.0 )
install(TARGETS ${LIBNAME} 	RUNTIME DESTINATION bin
						LIBRARY DESTINATION lib 
						ARCHIVE DESTINATION lib)
install(DIRECTORY ../yuri DESTINATION include
          				FILES_MATCHING PATTERN "*.h")
               

						