# Set name of the module
SET (MODULE sdl_window)

# Set all source files module uses
SET (SRC SDLWindow.cpp
		 SDLWindow.h)


 
add_library(${MODULE} MODULE ${SRC})
include_directories(${SDL_INCLUDE_DIR})
target_link_libraries(${MODULE} ${LIBNAME} ${SDL_LIBRARY})

YURI_INSTALL_MODULE(${MODULE})
