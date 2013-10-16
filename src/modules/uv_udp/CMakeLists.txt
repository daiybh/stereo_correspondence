# Set name of the module
SET (MODULE uv_udp)

# Set all source files module uses
SET (SRC UVUdpSocket.cpp
		 UVUdpSocket.h
		 ${UV_BASE}/rtp/net_udp.c
		 ${UV_BASE}/debug.c
		 ${UV_BASE}/host.c)



include_directories(${UV_BASE})
 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})
SET_TARGET_PROPERTIES(uv_udp PROPERTIES COMPILE_FLAGS -DHAVE_CONFIG_H)
YURI_INSTALL_MODULE(${MODULE})
