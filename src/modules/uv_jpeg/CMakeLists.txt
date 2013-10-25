# Set name of the module
SET (MODULE uv_jpeg)

# Set all source files module uses
SET (SRC UVJpegCompress.cpp
		 UVJpegCompress.h
		 ${UV_BASE}/video_compress/jpeg.c)


include_directories(${UV_BASE} ${UV_BASE}/..) 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${LIBNAME_HELPER_UV} ${JPEG_LIBRARY})
SET_TARGET_PROPERTIES(${MODULE} PROPERTIES COMPILE_FLAGS -DHAVE_CONFIG_H)

YURI_INSTALL_MODULE(${MODULE})
