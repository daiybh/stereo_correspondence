SET (MODULE v4l2)
CHECK_INCLUDE_FILES (linux/videodev2.h HAVE_VIDEODEV2_H)

SET (SRC V4l2Source.cpp
		 V4l2Source.h)



# You shouldn't need to edit anything below this line 
SET (MODULE_LIB yuri2_module_${MODULE})
add_library(${MODULE_LIB} MODULE ${SRC})
target_link_libraries(${MODULE_LIB} ${LIBNAME})
install(TARGETS ${MODULE_LIB} LIBRARY DESTINATION ${TARGET_MODULE_PATH}) 

add_custom_command(TARGET ${MODULE_LIB} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE_DIR:${MODULE_LIB}>/$<TARGET_FILE_NAME:${MODULE_LIB}>"
    "${MODULE_DIR}/$<TARGET_FILE_NAME:${MODULE_LIB}>"
    )						