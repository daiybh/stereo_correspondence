# Set name of the module
SET (MODULE ${module})

# Set all source files module uses
SET (SRC ${class_name}.cpp
		 ${class_name}.h)


 
add_library($${MODULE} MODULE $${SRC})
target_link_libraries($${MODULE} $${LIBNAME})

YURI_INSTALL_MODULE($${MODULE})
