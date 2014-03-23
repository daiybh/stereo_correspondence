# Set name of the module
SET (MODULE shout_source)

# Set all source files module uses
SET (SRC ShoutSource.cpp
		 ShoutSource.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
