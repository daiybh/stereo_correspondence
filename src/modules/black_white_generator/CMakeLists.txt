# Set name of the module
SET (MODULE black_white_generator)

# Set all source files module uses
SET (SRC BlackWhiteGenerator.cpp
		 BlackWhiteGenerator.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
