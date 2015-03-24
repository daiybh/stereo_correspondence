# Set name of the module
SET (MODULE sync_frames)

# Set all source files module uses
SET (SRC SyncFrames.cpp
		 SyncFrames.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
