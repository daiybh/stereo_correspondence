# Set name of the module
SET (MODULE tsmuxer)

# Set all source files module uses
SET (SRC TSMuxer.cpp
		 TSMuxer.h)



# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${LIBNAME_HELPER_LIBAV} ${libavformat_LIBRARIES})

YURI_INSTALL_MODULE(${MODULE})