find_package( Boost 1.46 COMPONENTS program_options REQUIRED)



add_executable(yuri2 yuri.cpp)
target_link_libraries ( yuri2 ${LIBNAME} ${Boost_LIBRARIES} )
install(TARGETS yuri2 RUNTIME DESTINATION bin)

add_custom_command(TARGET yuri2 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE_DIR:yuri2>/$<TARGET_FILE_NAME:yuri2>
    "${BIN_DIR}/$<TARGET_FILE_NAME:yuri2>"
    )