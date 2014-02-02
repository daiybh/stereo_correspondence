# Set name of the module
SET (MODULE ultragrid)

include_directories(${UV_BASE} ${UV_BASE}/..)

SET(SRC register.cpp)

find_package(ALSA)
find_package(GLEW)
find_package(GLUT)
find_package(SDL)
find_package(X11)

IF(NOT YURI_DISABLE_LIBAV)
	pkg_check_modules(libavcodec libavcodec)
ENDIF()

# ALSA
IF(ALSA_FOUND)
    SET (SRC ${SRC}
             UVAlsaInput.cpp
    		 UVAlsaInput.h
    		 UVAlsaOutput.cpp
    		 UVAlsaOutput.h
    		 ${UV_BASE}/audio/capture/alsa.c
    		 ${UV_BASE}/audio/playback/alsa.c)
    
    SET (LIBS ${ALSA_LIBRARIES})
    add_definitions(-DYURI_UV_ALSA_SUPPORTED)
ENDIF()

# AudioTestcard
SET (SRC ${SRC} 
         UVAudioTestcard.cpp
		 UVAudioTestcard.h
		 ${UV_BASE}/audio/capture/testcard.c)

# UVConvert

SET (SRC ${SRC}
         UVConvert.cpp
		 UVConvert.h)

# UVGl
IF(GLUT_FOUND AND GLEW_FOUND)
    SET (SRC ${SRC}
            UVGl.cpp
    		 UVGl.h
    		 ${UV_BASE}/gl_context.c
    		 ${UV_BASE}/glx_common.c
    		 ${UV_BASE}/video_display/gl.c
    		 )
    SET(LIBS ${LIBS}  ${GLUT_LIBRARY} ${GLEW_LIBRARY})
    add_definitions(-DYURI_UV_GLUT_SUPPORTED)
ENDIF()

#UV jpeg
IF(GPUJPEG_FOUND)
    SET (SRC ${SRC}
             UVJpegCompress.cpp
    		 UVJpegCompress.h
    		 ${UV_BASE}/video_compress/jpeg.c)
    		 
    SET(LIBS ${LIBS}  ${JPEG_LIBRARY})
    add_definitions(-DYURI_UV_JPEG_SUPPORTED)
ENDIF()

# libav
IF(libavcodec_FOUND)
    SET (SRC ${SRC} 
             UVLibav.cpp
    		 UVLibav.h
    		 ${UV_BASE}/video_compress/libavcodec.c
    #		 ${UV_BASE}/gl_context.c
    #		 ${UV_BASE}/glx_common.c
    		 )
    SET(LIBS ${LIBS} ${libavcodec_LIBRARIES})
    add_definitions(-DYURI_UV_LIBAV_SUPPORTED)
ENDIF()

# RTDXT
IF(GLEW_FOUND)
    SET (SRC ${SRC}
             UVRTDxt.cpp
    		 UVRTDxt.h
    		 ${UV_BASE}/video_compress/dxt_glsl.c
    		 ${UV_BASE}/gl_context.c
    		 ${UV_BASE}/glx_common.c
    		 ${UV_BASE}/../dxt_compress/dxt_encoder.c
    		 ${UV_BASE}/../dxt_compress/dxt_util.c
    		 )
    		 
    SET(LIBS ${LIBS} ${GLEW_LIBRARY})
    add_definitions(-DYURI_UV_RTDXT_SUPPORTED)
ENDIF()

# RTP
SET (SRC ${SRC}
         UVRtpReceiver.h
         UVRtpReceiver.cpp 
         UVRtpSender.cpp
		 UVRtpSender.h
		 )
		 
# Screen
if (X11_FOUND)
    SET (SRC ${SRC}
             UVScreen.cpp
    	     UVScreen.h
    	     ${UV_BASE}/video_capture/screen_x11.c)
    add_definitions(-DYURI_UV_SCREEN_SUPPORTED)
ENDIF()

 # SDL
IF(SDL_FOUND)
    SET(SRC ${SRC}
             UVSdl.cpp
    		 UVSdl.h
    		 ${UV_BASE}/video_display/sdl.c)
    
    SET(LIBS ${LIBS} ${SDL_LIBRARY})
    add_definitions(-DYURI_UV_SDL_SUPPORTED)
ENDIF()

# testcard
SET (SRC ${SRC}
         UVTestcard.cpp
		 UVTestcard.h
		 ${UV_BASE}/video_capture/testcard.c
		 ${UV_BASE}/video_capture/testcard_common.c)

# UDP socket
SET (SRC ${SRC} 
         UVUdpSocket.cpp
		 UVUdpSocket.h
		 )

# UYVY
SET (SRC ${SRC}
         UVUyvy.cpp
		 UVUyvy.h
		 ${UV_BASE}/video_compress/uyvy.c)
		 
# V4L2
pkg_check_modules(libv4l2 libv4l2)
IF (libv4l2_FOUND)
    SET (SRC ${SRC}
             UVV4l2.cpp
    		 UVV4l2.h
    		 ${UV_BASE}/video_capture/v4l2.c)
    SET(LIBS ${LIBS} ${libv4l2_LIBRARIES})
ENDIF()
		 

add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${LIBNAME_HELPER_UV} ${LIBS})
SET_TARGET_PROPERTIES(${MODULE} PROPERTIES COMPILE_FLAGS -DHAVE_CONFIG_H)

YURI_INSTALL_MODULE(${MODULE})

# END of file
# ================================
