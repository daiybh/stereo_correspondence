SET(MODULE "read_pcap")

SET(SRC ReadPcap.cpp
		ReadPcap.h)
		
 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})