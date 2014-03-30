# Set name of the module
SET (MODULE x265_module)

# Set all source files module uses
SET (SRC X265Encoder.cpp
		 X265Encoder.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${X265_LIBRARIES})

YURI_INSTALL_MODULE(${MODULE})
