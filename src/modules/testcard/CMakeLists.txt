# Set name of the module
SET (MODULE testcard)

# Set all source files module uses
SET (SRC TestCard.cpp
		 TestCard.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
