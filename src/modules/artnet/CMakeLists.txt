# Set name of the module
SET (MODULE artnet)

# Set all source files module uses
SET (SRC ArtNet.cpp
		 ArtNet.h
		 ArtNetPacket.cpp
		 ArtNetPacket.h
		 )


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${Boost_REGEX_LIBRARY})

YURI_INSTALL_MODULE(${MODULE})
