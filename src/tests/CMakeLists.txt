include( CTest )

include_directories ( ${Boost_INCLUDE_DIRS} )
link_directories (${Boost_LIBRARY_DIRS})

add_library(yuri_test_main STATIC test_main.cpp)

add_executable(yuri_test_suite  test_base.cpp
								time_tests.cpp
								test_irange.cpp
								test_make_list.cpp
								test_parameters.cpp
								test_filesystem.cpp
								test_string_generator.cpp
								test_string.cpp
								test_color.cpp
								test_any.cpp
								
								test_state_table.cpp
								)
								
target_link_libraries (yuri_test_suite yuri_test_main ${LIBNAME})

add_executable(yuri_test_register test_register.cpp)
								
target_link_libraries (yuri_test_register yuri_test_main ${LIBNAME})


add_test (core_test ${EXECUTABLE_OUTPUT_PATH}/yuri_test_suite )
add_test (register_test ${EXECUTABLE_OUTPUT_PATH}/yuri_test_register )