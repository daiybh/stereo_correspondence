# Set name of the module
SET (MODULE uv_gl)

# Set all source files module uses
SET (SRC UVGl.cpp
		 UVGl.h
		 ${UV_BASE}/gl_context.c
		 ${UV_BASE}/glx_common.c
		 ${UV_BASE}/video_display/gl.c
		 )


include_directories(${UV_BASE} ${UV_BASE}/..) 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${LIBNAME_HELPER_UV} ${GLUT_LIBRARY} ${GLEW_LIBRARY})
SET_TARGET_PROPERTIES(${MODULE} PROPERTIES COMPILE_FLAGS -DHAVE_CONFIG_H)


YURI_INSTALL_MODULE(${MODULE})
