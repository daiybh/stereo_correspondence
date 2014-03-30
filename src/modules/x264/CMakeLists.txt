# Set name of the module
SET (MODULE x264_module)

# Set all source files module uses
SET (SRC X264Encoder.cpp
		 X264Encoder.h
		 register.cpp)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${X264_LIBRARIES}) 

YURI_INSTALL_MODULE(${MODULE})
