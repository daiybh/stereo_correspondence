# Set name of the module
SET (MODULE theora_module)

# Set all source files module uses
SET (SRC TheoraEncoder.cpp
		 TheoraEncoder.h
		 register.cpp)


include_directories(${THEORAENC_INCLUDE_DIRS}) 
link_directories(${THEORAENC_LIBRARY_DIRS})
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${THEORAENC_LIBRARIES})

YURI_INSTALL_MODULE(${MODULE})
