# Set name of the module
SET (MODULE merge_frames)

# Set all source files module uses
SET (SRC 
	MergeFrames.cpp
	MergeFrames.h)

add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})