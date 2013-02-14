SET (MODULE yuri_convert)

SET (SRC YuriConvert.cpp
		 YuriConvert.h
		 )
SET(LINK ${LIBNAME})
IF(${CORE_CUDA})
	add_definitions("-DYURI_HAVE_CUDA")
	SET (SRC ${SRC}
		 YuriConvertCuda.cu)
	cuda_add_module(${MODULE} ${SRC})
	SET(LINK ${LINK} ${LIBNAME_HELPER_CUDA})
ELSE()
	add_library(${MODULE} MODULE ${SRC})	
ENDIF()

# You shouldn't need to edit anything below this line 

target_link_libraries(${MODULE} ${LINK})
install(TARGETS ${MODULE} LIBRARY DESTINATION ${TARGET_MODULE_PATH}) 

add_custom_command(TARGET ${MODULE} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE_DIR:${MODULE}>/$<TARGET_FILE_NAME:${MODULE}>"
    "${MODULE_DIR}/$<TARGET_FILE_NAME:${MODULE}>"
    )										
