# Set name of the module
SET (MODULE sage_output)

# Set all source files module uses
SET (SRC SageOutput.cpp
		 SageOutput.h)



# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
include_directories(${SAGE_INCLUDE_DIRS})
target_link_libraries(${MODULE} ${LIBNAME} ${SAGE_LIBRARY})

YURI_INSTALL_MODULE(${MODULE})