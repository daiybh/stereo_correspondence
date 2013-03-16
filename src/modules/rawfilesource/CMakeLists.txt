# Set name of the module
SET (MODULE rawfilesource)

# Set all source files module uses
SET (SRC RawFileSource.cpp
		 RawFileSource.h)



# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${Boost_REGEX_LIBRARY})

YURI_INSTALL_MODULE(${MODULE})