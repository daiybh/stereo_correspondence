# Set name of the module
SET (MODULE freetype)

# Set all source files module uses
SET (SRC RenderText.cpp
		 RenderText.h)


 
include_directories( ${FREETYPE_INCLUDE_DIR_freetype2} )
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${FREETYPE_LIBRARY})

YURI_INSTALL_MODULE(${MODULE})
