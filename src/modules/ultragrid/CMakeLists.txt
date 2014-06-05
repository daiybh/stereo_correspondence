# Set name of the module
SET (MODULE ultragrid)

include_directories(${UV_BASE} ${UV_BASE}/..)

find_package(ALSA)
find_package(GLEW)
find_package(GLUT)
find_package(SDL)
find_package(X11)
find_package(OpenSSL)
pkg_check_modules(SPEEX speex speexdsp)

SET(SRC 
    YuriUltragrid.cpp
    YuriUltragrid.h
	UVVideoSink.cpp
	UVVideoSink.h
	UVVideoSource.cpp
	UVVideoSource.h
	UVVideoCompress.cpp
	UVVideoCompress.h
	
        #${UV_BASE}/capture_filter.c
	${UV_BASE}/debug.c
	${UV_BASE}/host.c
	${UV_BASE}/lib_common.c
#	${UV_BASE}/messaging.cpp
	#${UV_BASE}/module.c
	${UV_BASE}/ntp.c
	${UV_BASE}/pdb.c
	${UV_BASE}/tfrc.c
        #${UV_BASE}/tile.c
	${UV_BASE}/transmit.c
	${UV_BASE}/tv.c
	${UV_BASE}/video.c
	${UV_BASE}/video_capture.c
	${UV_BASE}/video_codec.c
        ${UV_BASE}/video_compress.cpp
	${UV_BASE}/video_decompress.c
	${UV_BASE}/video_display.c
	#${UV_BASE}/video_frame.c
	${UV_BASE}/x11_common.c
	
        #${UV_BASE}/audio/audio.c
	${UV_BASE}/audio/audio_capture.c
	${UV_BASE}/audio/audio_playback.c
	${UV_BASE}/audio/codec.c
        #${UV_BASE}/audio/echo.c
	${UV_BASE}/audio/export.c
        #${UV_BASE}/audio/resampler.c
	${UV_BASE}/audio/utils.c
	${UV_BASE}/audio/wav_reader.c
	
	${UV_BASE}/audio/capture/none.c
	${UV_BASE}/audio/capture/sdi.c
	
	${UV_BASE}/audio/codec/dummy_pcm.c
	
	${UV_BASE}/audio/playback/none.c
	${UV_BASE}/audio/playback/sdi.c
	
	
	${UV_BASE}/capture_filter/blank.c
	#${UV_BASE}/capture_filter/none.c
	${UV_BASE}/capture_filter/every.c
	#${UV_BASE}/capture_filter/logo.c
	
	${UV_BASE}/compat/platform_spin.c
	
	${UV_BASE}/crypto/md5.c
	#${UV_BASE}/crypto/crc_32.c
	${UV_BASE}/crypto/crypt_aes.c
	${UV_BASE}/crypto/crypt_aes_impl.c
	${UV_BASE}/crypto/crypt_des.c
	${UV_BASE}/crypto/random.c
	
	${UV_BASE}/module.c

	${UV_BASE}/rtp/audio_decoders.c
	${UV_BASE}/rtp/video_decoders.cpp
	${UV_BASE}/rtp/ldgm.cpp
	${UV_BASE}/rtp/net_udp.c
	${UV_BASE}/rtp/pbuf.c
	
	#${UV_BASE}/rtp/pc.cpp
        #${UV_BASE}/rtp/ll.cpp
	${UV_BASE}/rtp/rtp.c
        ${UV_BASE}/rtp/rtpenc_h264.c
	${UV_BASE}/rtp/rtp_callback.c
	
	#${UV_BASE}/utils/config_file.c
	${UV_BASE}/utils/list.c
	#${UV_BASE}/utils/message_queue.cpp
	${UV_BASE}/utils/resource_manager.cpp
	${UV_BASE}/utils/packet_counter.cpp
	${UV_BASE}/utils/ring_buffer.c
	${UV_BASE}/utils/worker.cpp
	
        #${UV_BASE}/vo_postprocess.c

	${UV_BASE}/video_frame.c
	
	${UV_BASE}/video_capture/null.c
	
	${UV_BASE}/video_compress/none.c
	
	${UV_BASE}/video_decompress/null.c
	
	${UV_BASE}/video_display/null.c
		
	${UV_BASE}/../ldgm-coding/ldgm-session.cpp
	${UV_BASE}/../ldgm-coding/ldgm-session-cpu.cpp
	${UV_BASE}/../ldgm-coding/tanner.cpp
	${UV_BASE}/../ldgm-coding/matrix-gen/ldpc-matrix.cpp
	${UV_BASE}/../ldgm-coding/matrix-gen/matrix-generator.cpp
)
SET(LIBS )
if (X11_FOUND)
    SET(SRC, ${SRC} 
             ${UV_BASE}/x11_common.c)
    SET(LIBS ${LIBS} ${X11_LIBRARIES})
    #SET(LIBS ${LIBS} ${X11_Xfixes_LIB})
endif()

IF(OPENSSL_FOUND)
    SET(SRC ${SRC}
    	#${UV_BASE}/crypto/openssl_encrypt.c
    	#${UV_BASE}/crypto/openssl_decrypt.c
	)
	SET(LIBS ${LIBS} ${OPENSSL_LIBRARIES})
ENDIF()
IF(SPEEX_FOUND)
    SET(LIBS ${LIBS} ${SPEEX_LIBRARIES} ${SPEEXDSP_LIBRARIES})
ENDIF()

SET(SRC ${SRC} register.cpp)

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
    
    SET (LIBS ${LIBS} ${ALSA_LIBRARIES})
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
    		 ${UV_BASE}/video_compress/dxt_glsl.cpp
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
		 ${UV_BASE}/video_compress/uyvy.cpp)
		 
# V4L2
pkg_check_modules(libv4l2 libv4l2)
IF (libv4l2_FOUND)
    SET (SRC ${SRC}
             UVV4l2.cpp
    		 UVV4l2.h
    		 ${UV_BASE}/video_capture/v4l2.c)
    SET(LIBS ${LIBS} ${libv4l2_LIBRARIES})
    add_definitions(-DYURI_UV_V4L2_SUPPORTED)
ENDIF()
		 

add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${LIBS})
SET_TARGET_PROPERTIES(${MODULE} PROPERTIES COMPILE_FLAGS "-DHAVE_CONFIG_H -DUV_IN_YURI -DBUILD_LIBRARIES")
YURI_INSTALL_MODULE(${MODULE})

# END of file
# ================================
