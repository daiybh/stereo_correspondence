# Set name of the module
SET (MODULE fade)

# Set all source files module uses
SET (SRC Fade.cpp
		 Fade.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
