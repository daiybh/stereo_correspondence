# Set name of the module
SET (MODULE uv_alsa_input)

# Set all source files module uses
SET (SRC UVAlsaInput.cpp
		 UVAlsaInput.h
		 UVAlsaOutput.cpp
		 UVAlsaOutput.h
		 register.cpp
		 ${UV_BASE}/audio/capture/alsa.c
		 ${UV_BASE}/audio/playback/alsa.c)


include_directories(${UV_BASE})

add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${LIBNAME_HELPER_UV} ${ALSA_LIBRARIES})
SET_TARGET_PROPERTIES(${MODULE} PROPERTIES COMPILE_FLAGS -DHAVE_CONFIG_H)

YURI_INSTALL_MODULE(${MODULE})
