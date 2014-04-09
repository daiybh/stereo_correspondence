# Set name of the module
SET (MODULE scale)

# Set all source files module uses
SET (SRC Scale.cpp
		 Scale.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
