# Set name of the module
SET (MODULE dimenco_out)

# Set all source files module uses
SET (SRC DimencoOut.cpp
		 DimencoOut.h)



# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
