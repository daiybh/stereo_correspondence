# Set name of the module
SET (MODULE overlay)

# Set all source files module uses
SET (SRC Overlay.cpp
		 Overlay.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
