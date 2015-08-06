include( CTest )

include_directories ( ${Boost_INCLUDE_DIRS} )
link_directories (${Boost_LIBRARY_DIRS})

add_executable(yuri_test_suite test_base.cpp
								time_tests.cpp
								test_irange.cpp
								test_make_list.cpp
								test_parameters.cpp
								test_filesystem.cpp
								test_string_generator.cpp
								test_string.cpp
								test_color.cpp)
target_link_libraries (yuri_test_suite ${LIBNAME})

add_executable(yuri_test_register test_register.cpp)
target_link_libraries (yuri_test_register ${LIBNAME})


add_test (core_test ${EXECUTABLE_OUTPUT_PATH}/yuri_test_suite )
add_test (register_test ${EXECUTABLE_OUTPUT_PATH}/yuri_test_register )