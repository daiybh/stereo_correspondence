# Set name of the module
SET (MODULE midi_device)

# Set all source files module uses
SET (SRC MidiDevice.cpp
		 MidiDevice.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${ALSA_LIBRARIES})

YURI_INSTALL_MODULE(${MODULE})
