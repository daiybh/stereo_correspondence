SET (YURI_LINK )
SET (YURI_LIBS )
SET (YURI_INCLUDE )
SET (YURI_SRC )

find_package( Boost 1.46 COMPONENTS thread system filesystem REQUIRED)
find_package( X11  REQUIRED)
find_package( OpenGL REQUIRED)
CHECK_INCLUDE_FILES (linux/videodev2.h HAVE_VIDEODEV2_H)
CHECK_INCLUDE_FILES (stdint.h HAVE_STDINT_H)

IF (NOT HAVE_STDINT_H)
	MESSAGE(FATAL_ERROR "Missing stdint.h. Please update your compile chain")
ENDIF()


SET (YURI_LINK ${YURI_LINK} 
		${Boost_LIBRARY_DIRS} 
		${X11_LIBRARY_DIRS})
		
SET (YURI_LIBS ${YURI_LIBS} 
		${Boost_LIBRARIES}  
		${X11_LIBRARIES}
		${OPENGL_gl_LIBRARY})
SET (YURI_INCLUDE ${YURI_INCLUDE} 
		${Boost_INCLUDE_DIRS}  
		${X11_INCLUDE_DIRS}
		)
SET (YURI_SRC ${YURI_SRC} )
add_definitions("-DYURI_MODULE_IN_TREE")

add_subdirectory(log)
add_subdirectory(config)
add_subdirectory(threads)
add_subdirectory(graphics)
add_subdirectory(exception)
add_subdirectory(io)

add_library(${LIBNAME} ${YURI_SRC})
link_directories ( ${YURI_LINK} )
include_directories ( ${YURI_INCLUDE} )
IF(UNIX)
target_link_libraries ( ${LIBNAME} ${YURI_LIBS} rt)
ENDIF()
SET_TARGET_PROPERTIES( ${LIBNAME} PROPERTIES SOVERSION 2 VERSION 2.5.0 )
install(TARGETS ${LIBNAME} 	RUNTIME DESTINATION bin
						LIBRARY DESTINATION lib 
						ARCHIVE DESTINATION lib)
               

						