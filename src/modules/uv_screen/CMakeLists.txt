# Set name of the module
SET (MODULE uv_screen)

# Set all source files module uses
SET (SRC UVScreen.cpp
		 UVScreen.h
		 ${UV_BASE}/video_capture/screen_x11.c)

include_directories(${UV_BASE})

add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${LIBNAME_HELPER_UV})
SET_TARGET_PROPERTIES(${MODULE} PROPERTIES COMPILE_FLAGS -DHAVE_CONFIG_H)

YURI_INSTALL_MODULE(${MODULE})
