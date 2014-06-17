# Set name of the module
SET (MODULE audio_latency)

# Set all source files module uses
SET (SRC AudioLatency.cpp
		 AudioLatency.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
