# Set name of the module
SET (MODULE json_config)

# Set all source files module uses
SET (SRC JsonConfig.cpp
		 JsonConfig.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${JSONCPP_LIBRARY})

YURI_INSTALL_MODULE(${MODULE})
