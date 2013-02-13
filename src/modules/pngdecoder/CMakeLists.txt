# Set name of the module
SET (MODULE pngdecoder)

# Set all source files module uses
SET (SRC PNGDecoder.cpp
		 PNGDecoder.h)


# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
include_directories(${PNG_INCLUDE_DIR})
add_definitions(${PNG_DEFINITIONS})
target_link_libraries(${MODULE} ${LIBNAME} ${PNG_LIBRARY})
install(TARGETS ${MODULE} LIBRARY DESTINATION ${TARGET_MODULE_PATH}) 

add_custom_command(TARGET ${MODULE} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE_DIR:${MODULE}>/$<TARGET_FILE_NAME:${MODULE}>"
    "${MODULE_DIR}/$<TARGET_FILE_NAME:${MODULE}>"
    )						