# Set name of the module
SET (MODULE png_module)

# Set all source files module uses
SET (SRC PngEncoder.cpp
		 PngEncoder.h
		 PngDecoder.cpp
		 PngDecoder.h
		 register.cpp)


include_directories(${PNG_PNG_INCLUDE_DIR}) 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${PNG_LIBRARIES})

YURI_INSTALL_MODULE(${MODULE})
