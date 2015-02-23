# Set name of the module
SET (MODULE file_picker)

# Set all source files module uses
SET (SRC FilePicker.cpp
		 FilePicker.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${Boost_REGEX_LIBRARY})

YURI_INSTALL_MODULE(${MODULE})
