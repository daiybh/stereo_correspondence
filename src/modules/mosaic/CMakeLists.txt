# Set name of the module
SET (MODULE mosaic)

# Set all source files module uses
SET (SRC Mosaic.cpp
		 Mosaic.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
