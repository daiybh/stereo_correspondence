# Set name of the module
SET (MODULE uv_v4l2)

# Set all source files module uses
SET (SRC UVV4l2.cpp
		 UVV4l2.h
		 ${UV_BASE}/video_capture/v4l2.c)

include_directories(${UV_BASE})

add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${LIBNAME_HELPER_UV} ${libv4l2_LIBRARIES})
SET_TARGET_PROPERTIES(uv_v4l2 PROPERTIES COMPILE_FLAGS -DHAVE_CONFIG_H)

YURI_INSTALL_MODULE(${MODULE})
