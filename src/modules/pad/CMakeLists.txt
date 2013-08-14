# Set name of the module
SET (MODULE pad)

# Set all source files module uses
SET (SRC Pad.cpp
		 Pad.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
