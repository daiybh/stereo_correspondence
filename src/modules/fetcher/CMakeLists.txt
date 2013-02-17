# Set name of the module
SET (MODULE fetcher)

# Set all source files module uses
SET (SRC Fetcher.cpp
		 Fetcher.h)


# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${CURL_LIBRARY})

YURI_INSTALL_MODULE(${MODULE})