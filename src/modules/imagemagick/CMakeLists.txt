# Set name of the module
SET (MODULE imagemagick_source)

# Set all source files module uses
SET (SRC ImageMagickSource.cpp
		 ImageMagickSource.h)


include_directories(${ImageMagick_Magick++_INCLUDE_DIR})

# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${ImageMagick_Magick++_LIBRARY})
install(TARGETS ${MODULE} LIBRARY DESTINATION ${TARGET_MODULE_PATH}) 

add_custom_command(TARGET ${MODULE} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE_DIR:${MODULE}>/$<TARGET_FILE_NAME:${MODULE}>"
    "${MODULE_DIR}/$<TARGET_FILE_NAME:${MODULE}>"
    )						