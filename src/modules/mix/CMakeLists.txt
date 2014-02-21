# Set name of the module
SET (MODULE mix)

# Set all source files module uses
SET (SRC Mix.cpp
		 Mix.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
