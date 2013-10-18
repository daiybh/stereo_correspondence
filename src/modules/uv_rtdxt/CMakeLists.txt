# Set name of the module
SET (MODULE uv_rtdxt)

# Set all source files module uses
SET (SRC UVRTDxt.cpp
		 UVRTDxt.h
		 ${UV_BASE}/video_compress/dxt_glsl.c
		 ${UV_BASE}/gl_context.c
		 ${UV_BASE}/glx_common.c
		 ${UV_BASE}/../dxt_compress/dxt_encoder.c
		 ${UV_BASE}/../dxt_compress/dxt_util.c
		 )


 

include_directories(${UV_BASE} ${UV_BASE}/..) 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${LIBNAME_HELPER_UV} ${GLEW_LIBRARY})
SET_TARGET_PROPERTIES(${MODULE} PROPERTIES COMPILE_FLAGS -DHAVE_CONFIG_H)

YURI_INSTALL_MODULE(${MODULE})

