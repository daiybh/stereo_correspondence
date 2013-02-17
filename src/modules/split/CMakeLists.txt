# Set name of the module
SET (MODULE split)

# Set all source files module uses
SET (SRC Split.cpp
		 Split.h)



# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})