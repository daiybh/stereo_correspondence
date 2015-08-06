# Set name of the module
SET (MODULE sdl2_window)

# Set all source files module uses
SET (SRC SDL2Window.cpp
		 SDL2Window.h)


include_directories(${SDL2_INCLUDE_DIRS})
SET(LIBS ${LIBNAME} ${SDL2_LIBRARIES})

IF(NOT YURI_DISABLE_OPENGL)
    find_package(OpenGL)
    if (OPENGL_FOUND)
        add_definitions("-DYURI_SDL_OPENGL=1")
        SET(LIBS ${LIBS} ${LIBNAME_HELPER_GL})
    ENDIF()
ENDIF()

 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBS})

YURI_INSTALL_MODULE(${MODULE})
