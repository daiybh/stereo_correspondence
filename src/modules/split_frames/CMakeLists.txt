# Set name of the module
SET (MODULE split_frames)

# Set all source files module uses
SET (SRC 
	SplitFrames.cpp
	SplitFrames.h)

add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})