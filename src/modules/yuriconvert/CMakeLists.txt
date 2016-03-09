SET (MODULE yuri_convert)

SET (SRC YuriConvert.cpp
		 YuriConvert.h
		 )

SET(LINK ${LIBNAME})
IF(${CORE_CUDA} AND FALSE)
	add_definitions("-DYURI_HAVE_CUDA")
	SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
	list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-O2")
	CUDA_INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/src )

	SET (SRC ${SRC}
		 YuriConvertCuda.cu)
	cuda_add_module(${MODULE} ${SRC})
	SET(LINK ${LINK} ${LIBNAME_HELPER_CUDA})
ELSE()
	add_library(${MODULE} MODULE ${SRC})	
ENDIF()

# You shouldn't need to edit anything below this line 

target_link_libraries(${MODULE} ${LINK})

YURI_INSTALL_MODULE(${MODULE})