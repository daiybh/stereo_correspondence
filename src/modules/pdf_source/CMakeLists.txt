# Set name of the module
SET (MODULE pdf_source)

# Set all source files module uses
SET (SRC PdfSource.cpp
		 PdfSource.h)


include_directories(${POPPLERCPP_INCLUDE_DIRS})
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${POPPLERCPP_LIBRARIES})

YURI_INSTALL_MODULE(${MODULE})
