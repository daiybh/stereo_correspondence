# Set name of the module
SET (MODULE uv_rtp_sender)

# Set all source files module uses
SET (SRC UVRtpSender.cpp
		 UVRtpSender.h
		 		 
		 
		 )


include_directories(${UV_BASE} ${UV_BASE}/..) 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${LIBNAME_HELPER_UV})
SET_TARGET_PROPERTIES(${MODULE} PROPERTIES COMPILE_FLAGS -DHAVE_CONFIG_H)


YURI_INSTALL_MODULE(${MODULE})
