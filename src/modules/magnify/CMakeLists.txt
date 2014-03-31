# Set name of the module
SET (MODULE magnify)

# Set all source files module uses
SET (SRC Magnify.cpp
		 Magnify.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
