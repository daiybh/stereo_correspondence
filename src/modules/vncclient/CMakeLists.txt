# Set name of the module
SET (MODULE vncclient)

# Set all source files module uses
SET (SRC VNCClient.cpp
		 VNCClient.h)



# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(${MODULE} ${LIBNAME} ${LIBNAME_HELPER_ASIO})
install(TARGETS ${MODULE} LIBRARY DESTINATION ${TARGET_MODULE_PATH}) 

add_custom_command(TARGET ${MODULE} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE_DIR:${MODULE}>/$<TARGET_FILE_NAME:${MODULE}>"
    "${MODULE_DIR}/$<TARGET_FILE_NAME:${MODULE}>"
    )						