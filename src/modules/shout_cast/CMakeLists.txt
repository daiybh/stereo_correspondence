# Set name of the module
SET (MODULE shout_cast)

# Set all source files module uses
SET (SRC ShoutOutput.cpp
		 ShoutOutput.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${SHOUT_LIBRARIES})

YURI_INSTALL_MODULE(${MODULE})
