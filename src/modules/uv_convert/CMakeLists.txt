# Set name of the module
SET (MODULE uv_convert)

# Set all source files module uses
SET (SRC UVConvert.cpp
		 UVConvert.h)


include_directories(${UV_BASE})

add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${LIBNAME_HELPER_UV})
SET_TARGET_PROPERTIES(${MODULE} PROPERTIES COMPILE_FLAGS -DHAVE_CONFIG_H)

YURI_INSTALL_MODULE(${MODULE})
