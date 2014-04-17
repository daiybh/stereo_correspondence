# Set name of the module
SET (MODULE event_info)

# Set all source files module uses
SET (SRC register.cpp 
         EventInfo.cpp
		 EventInfo.h
		 PassEvent.cpp
		 PassEvent.h
		 )


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
