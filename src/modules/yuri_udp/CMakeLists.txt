# Set name of the module
SET (MODULE yuri_udp)

# Set all source files module uses
SET (SRC YuriUdp.cpp
		 YuriUdp.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
