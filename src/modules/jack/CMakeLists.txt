# Set name of the module
SET (MODULE jack_module)

# Set all source files module uses
SET (SRC JackOutput.cpp
		 JackOutput.h)


 
add_library(${MODULE} MODULE ${SRC})
include_directories(${JACK_INCLUDE_DIRS})
target_link_libraries(${MODULE} ${LIBNAME} ${JACK_LIBRARIES})

YURI_INSTALL_MODULE(${MODULE})
