# Set name of the module
SET (MODULE split_planes)

# Set all source files module uses
SET (SRC SplitPlanes.cpp
		 SplitPlanes.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
