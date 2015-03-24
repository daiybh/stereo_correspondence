# Set name of the module
SET (MODULE wav)

# Set all source files module uses
SET (SRC WaveDump.cpp
		 WaveDump.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
