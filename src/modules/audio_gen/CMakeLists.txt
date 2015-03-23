# Set name of the module
SET (MODULE audio_gen)

# Set all source files module uses
SET (SRC SineGenerator.cpp
		 SineGenerator.h
		 register.cpp)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
