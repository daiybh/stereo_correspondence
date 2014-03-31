# Set name of the module
SET (MODULE bayer)

# Set all source files module uses
SET (SRC BayerVisualize.cpp
		 BayerVisualize.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
