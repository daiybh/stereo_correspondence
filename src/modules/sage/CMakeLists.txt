# Set name of the module
SET (MODULE sage_output)

# Set all source files module uses
SET (SRC SageOutput.cpp
		 SageOutput.h)



# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
include_directories(${SAGE_INCLUDE_DIRS})
target_link_libraries(${MODULE} ${LIBNAME} ${SAGE_LIBRARY})
install(TARGETS ${MODULE} LIBRARY DESTINATION ${TARGET_MODULE_PATH}) 

add_custom_command(TARGET ${MODULE} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE_DIR:${MODULE}>/$<TARGET_FILE_NAME:${MODULE}>"
    "${MODULE_DIR}/$<TARGET_FILE_NAME:${MODULE}>"
    )						