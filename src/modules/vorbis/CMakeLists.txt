# Set name of the module
SET (MODULE vorbis_module)

# Set all source files module uses
SET (SRC VorbisEncoder.cpp
		 VorbisEncoder.h
		 register.cpp)


include_directories(${VORBISENC_INCLUDE_DIRS})
link_directories(${VORBISENC_LIBRARY_DIRS}) 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${VORBISENC_LIBRARIES})

YURI_INSTALL_MODULE(${MODULE})
