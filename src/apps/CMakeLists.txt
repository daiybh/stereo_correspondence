IF(NOT YURI_DISABLE_BOOST)
	find_package( Boost 1.46 COMPONENTS program_options regex)
	IF(Boost_PROGRAM_OPTIONS_FOUND)
		add_definitions(-DHAVE_BOOST_PROGRAM_OPTIONS)
	ENDIF()
ENDIF()

include_directories ( ${Boost_INCLUDE_DIRS})
link_directories (${Boost_LIBRARY_DIRS})

add_executable(yuri2 	yuri/yuri_listings.cpp 
						yuri/yuri_listings.h
						yuri/try_conversion.cpp
						yuri/try_conversion.h
						yuri2.cpp)

IF(WIN32)
target_link_libraries ( yuri2 ${LIBNAME})
ELSE()
target_link_libraries ( yuri2 ${LIBNAME} ${Boost_LIBRARIES} )
ENDIF()
install(TARGETS yuri2 RUNTIME DESTINATION bin)

add_executable(yuri_test_module test_module.cpp)
target_link_libraries (yuri_test_module ${LIBNAME} )
install(TARGETS yuri_test_module RUNTIME DESTINATION bin)

#add_library(yuri_wrapper MODULE wrapper.cpp)
#target_link_libraries (yuri_wrapper ${LIBNAME} )
#YURI_INSTALL_EXE_LIB(yuri_wrapper)

IF(Boost_REGEX_FOUND)
add_executable(yuri_simple 	
						yuri_simple.cpp
						simple/SimpleBuilder.h
						simple/SimpleBuilder.cpp)
target_link_libraries ( yuri_simple ${LIBNAME} ${Boost_REGEX_LIBRARY})
ENDIF()