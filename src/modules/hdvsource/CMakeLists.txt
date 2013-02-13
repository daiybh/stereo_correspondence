# Set name of the module
SET (MODULE hdvsource)

# Set all source files module uses
SET (SRC HDVSource.cpp
		 HDVSource.h)

# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
include_directories(${ieee1394_INCLUDE_DIR})
target_link_libraries(${MODULE} ${LIBNAME} ${LIBNAME_HELPER_IEEE1394})
install(TARGETS ${MODULE} LIBRARY DESTINATION ${TARGET_MODULE_PATH}) 

add_custom_command(TARGET ${MODULE} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE_DIR:${MODULE}>/$<TARGET_FILE_NAME:${MODULE}>"
    "${MODULE_DIR}/$<TARGET_FILE_NAME:${MODULE}>"
    )						