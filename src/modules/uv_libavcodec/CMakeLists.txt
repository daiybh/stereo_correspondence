# Set name of the module
SET (MODULE uv_libav)

# Set all source files module uses
SET (SRC UVLibav.cpp
		 UVLibav.h
		 ${UV_BASE}/video_compress/libavcodec.c
		 
#		 ${UV_BASE}/gl_context.c
#		 ${UV_BASE}/glx_common.c
		 )


 

include_directories(${UV_BASE} ${UV_BASE}/..) 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${LIBNAME_HELPER_UV} ${libavcodec_LIBRARIES})
SET_TARGET_PROPERTIES(${MODULE} PROPERTIES COMPILE_FLAGS -DHAVE_CONFIG_H)

YURI_INSTALL_MODULE(${MODULE})

