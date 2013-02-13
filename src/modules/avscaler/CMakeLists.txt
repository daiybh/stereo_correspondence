# Set name of the module
SET (MODULE avscaler)

# Set all source files module uses
SET (SRC AVScaler.cpp
		 AVScaler.h)



# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME}  ${LIBNAME_HELPER_LIBAV} ${libswscale_LIBRARIES})
install(TARGETS ${MODULE} LIBRARY DESTINATION ${TARGET_MODULE_PATH}) 

add_custom_command(TARGET ${MODULE} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE_DIR:${MODULE}>/$<TARGET_FILE_NAME:${MODULE}>"
    "${MODULE_DIR}/$<TARGET_FILE_NAME:${MODULE}>"
    )						