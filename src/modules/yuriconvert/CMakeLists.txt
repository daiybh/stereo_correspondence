SET (MODULE yuri_convert)

SET (SRC YuriConvert.cpp
		 YuriConvert.h
		 )
<<<<<<< Upstream, based on master
=======

>>>>>>> 341e532 Removed link dependency on libyuri_helper_cuda from yuri_convert when compiled without cuda support
SET(LINK ${LIBNAME})
IF(${CORE_CUDA})
	add_definitions("-DYURI_HAVE_CUDA")
	SET (SRC ${SRC}
		 YuriConvertCuda.cu)
	cuda_add_module(${MODULE} ${SRC})
<<<<<<< Upstream, based on master
	SET(LINK ${LINK} ${LIBNAME_HELPER_CUDA})
=======
	SET (LINK ${LINK} ${LIBNAME_HELPER_CUDA})
>>>>>>> 341e532 Removed link dependency on libyuri_helper_cuda from yuri_convert when compiled without cuda support
ELSE()
	add_library(${MODULE} MODULE ${SRC})	
ENDIF()

# You shouldn't need to edit anything below this line 

<<<<<<< Upstream, based on master
target_link_libraries(${MODULE} ${LINK})
=======
target_link_libraries(${MODULE} ${LINK} )
>>>>>>> 341e532 Removed link dependency on libyuri_helper_cuda from yuri_convert when compiled without cuda support
install(TARGETS ${MODULE} LIBRARY DESTINATION ${TARGET_MODULE_PATH}) 

add_custom_command(TARGET ${MODULE} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE_DIR:${MODULE}>/$<TARGET_FILE_NAME:${MODULE}>"
    "${MODULE_DIR}/$<TARGET_FILE_NAME:${MODULE}>"
    )										
