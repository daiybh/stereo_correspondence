# Set name of the module
SET (MODULE frei0r)

# Set all source files module uses
SET (SRC Frei0rBase.cpp
		 Frei0rBase.h 
	     Frei0rWrapper.cpp
		 Frei0rWrapper.h
		 Frei0rSource.cpp
		 Frei0rSource.h
		 Frei0rMixer.cpp
		 Frei0rMixer.h
		 frei0r_module.h
		 frei0r_register.cpp)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
