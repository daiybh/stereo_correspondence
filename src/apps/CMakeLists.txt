find_package( Boost 1.46 COMPONENTS program_options REQUIRED)



add_executable(yuri2 yuri.cpp)
target_link_libraries ( yuri2 ${LIBNAME} ${Boost_LIBRARIES} )
install(TARGETS yuri2 RUNTIME DESTINATION bin)

add_executable(yuri_test_module test_module.cpp)
target_link_libraries (yuri_test_module ${LIBNAME} )
install(TARGETS yuri_test_module RUNTIME DESTINATION bin)
