find_package( Boost 1.46 COMPONENTS program_options REQUIRED)

include_directories ( ${Boost_INCLUDE_DIRS})
link_directories (${Boost_LIBRARY_DIRS})
add_executable(yuri2 yuri.cpp)
IF(WIN32)
target_link_libraries ( yuri2 ${LIBNAME})
ELSE()
target_link_libraries ( yuri2 ${LIBNAME} ${Boost_LIBRARIES} )
ENDIF()
install(TARGETS yuri2 RUNTIME DESTINATION bin)

add_executable(yuri_test_module test_module.cpp)
target_link_libraries (yuri_test_module ${LIBNAME} )
install(TARGETS yuri_test_module RUNTIME DESTINATION bin)

add_library(yuri_wrapper MODULE wrapper.cpp)
target_link_libraries (yuri_wrapper ${LIBNAME} )
YURI_INSTALL_EXE_LIB(yuri_wrapper)
