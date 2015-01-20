# Set name of the module
SET (MODULE rawavfile)

# Set all source files module uses
SET (SRC RawAVFile.cpp
		 RawAVFile.h)



check_library_exists(avcodec av_frame_get_channels ${libavcodec_LIBDIR} HAVE_FRAME_GET_CHANNELS)
if (NOT HAVE_FRAME_GET_CHANNELS)
    add_definitions(-DBROKEN_FFMPEG)
endif()

# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME}  ${LIBNAME_HELPER_LIBAV} ${libavformat_LIBRARIES})

YURI_INSTALL_MODULE(${MODULE})