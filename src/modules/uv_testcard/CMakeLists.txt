# Set name of the module
SET (MODULE uv_testcard)

# Set all source files module uses
SET (SRC UVTestcard.cpp
		 UVTestcard.h
		 ${UV_BASE}/video_capture/testcard.c
		 ${UV_BASE}/video_capture/testcard_common.c)

include_directories(${UV_BASE})

add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${LIBNAME_HELPER_UV})
SET_TARGET_PROPERTIES(${MODULE} PROPERTIES COMPILE_FLAGS -DHAVE_CONFIG_H)

YURI_INSTALL_MODULE(${MODULE})
