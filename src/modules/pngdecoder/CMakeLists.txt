# Set name of the module
SET (MODULE pngdecoder)

# Set all source files module uses
SET (SRC PNGDecoder.cpp
		 PNGDecoder.h)


# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
include_directories(${PNG_INCLUDE_DIR})
add_definitions(${PNG_DEFINITIONS})
target_link_libraries(${MODULE} ${LIBNAME} ${PNG_LIBRARY})

YURI_INSTALL_MODULE(${MODULE})