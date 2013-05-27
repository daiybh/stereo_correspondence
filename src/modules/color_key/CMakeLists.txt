# Set name of the module
SET (MODULE color_key)

# Set all source files module uses
SET (SRC ColorKey.cpp
		 ColorKey.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
