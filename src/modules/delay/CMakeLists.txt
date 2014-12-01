# Set name of the module
SET (MODULE delay)

# Set all source files module uses
SET (SRC Delay.cpp
		 Delay.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
