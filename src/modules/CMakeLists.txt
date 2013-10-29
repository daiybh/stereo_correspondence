find_package( Boost 1.46 COMPONENTS regex)
include_directories( ${Boost_INCLUDE_DIRS})
link_directories (${Boost_LIBRARY_DIRS})



add_subdirectory(color_key)
add_subdirectory(combine)
add_subdirectory(diff)
add_subdirectory(dup)
add_subdirectory(event_info)
add_subdirectory(fade)
add_subdirectory(flip)
add_subdirectory(frame_info)
add_subdirectory(irc_client)
add_subdirectory(merge_frames)
add_subdirectory(null)
add_subdirectory(overlay)
add_subdirectory(pad)
add_subdirectory(rotate)
add_subdirectory(split_frames)
add_subdirectory(testcard)
add_subdirectory(yuriconvert)
add_subdirectory(osc_receiver)
add_subdirectory(vncclient)

IF(UNIX)
	add_subdirectory(yuri_tcp)
ENDIF()

CHECK_INCLUDE_FILES (linux/videodev2.h HAVE_VIDEODEV2_H)

IF(NOT YURI_DISABLE_LIBAV)
	pkg_check_modules(libavformat libavformat) 
	pkg_check_modules(libavutil libavutil)
	pkg_check_modules(libswscale libswscale)
ENDIF()

IF (CORE_DECKLINK)
	find_package(DECKLINK)
ENDIF()


IF(HAVE_VIDEODEV2_H)
	add_subdirectory(v4l2)
ENDIF()

find_package(SDL)
IF(SDL_FOUND)
	add_subdirectory(sdl_window)
ENDIF()

find_package(ALSA)
IF(ALSA_FOUND)
	add_subdirectory(midi_device)
	add_subdirectory(alsa_output)
ENDIF()

find_package(JPEG)
IF(JPEG_FOUND)
	add_subdirectory(jpegdecoder)
#	add_subdirectory(jpegencoder)
ENDIF()

IF(CORE_DECKLINK AND DECKLINK_FOUND)
	add_subdirectory(decklink)
ENDIF()

IF(${CORE_IEEE1394})
	add_subdirectory(dvsource)
	#add_subdirectory(hdvsource)
ENDIF()

#### UV #####
#IF(NOT YURI_DISABLE_ULTRAGRID)
	# This should be replaced by a proper detection!
	
IF(CORE_UV)
	SET(UV_BASE ${CMAKE_SOURCE_DIR}/../ultragrid/src)	
	add_subdirectory(uv_audio_testcard)
	add_subdirectory(uv_udp)
	add_subdirectory(uv_testcard)
	add_subdirectory(uv_rtp_sender)
	add_subdirectory(uv_rtp_receiver)
	
	
	
	find_package(X11)
	if (X11_FOUND)
		add_subdirectory(uv_screen)
	ENDIF()
	
	pkg_check_modules(libv4l2 libv4l2)
	IF (libv4l2_FOUND)
		add_subdirectory(uv_v4l2)
	ENDIF()
	
	IF(SDL_FOUND)
		add_subdirectory(uv_sdl)
	ENDIF()
	
	find_package(GLUT)
	find_package(GLEW)
	IF(GLUT_FOUND AND GLEW_FOUND)
		add_subdirectory(uv_gl)
		add_subdirectory(uv_rtdxt)
	ENDIF()
	
	IF(ALSA_FOUND)
		add_subdirectory(uv_alsa)
	ENDIF()
	
	pkg_check_modules(libavcodec libavcodec)
	IF(libavcodec_FOUND)
		add_subdirectory(uv_libavcodec)
	ENDIF()
	
	IF(GPUJPEG_FOUND)
		add_subdirectory(uv_jpeg)
	ENDIF()
ENDIF()
#############


IF(FALSE)
#################################################################
# Add modules that are compile unconditionally
#################################################################
add_subdirectory(anaglyph)
add_subdirectory(blank)
add_subdirectory(crop)

add_subdirectory(dummy)

add_subdirectory(filedump)


add_subdirectory(split)
add_subdirectory(dxt_compress)
add_subdirectory(temperature)
add_subdirectory(read_pcap)
add_subdirectory(fps_fixer)
add_subdirectory(ts_demuxer)


add_subdirectory(repack_audio)



add_subdirectory(fade)


#################################################################
# Let's check for libraries required 
# for optional modules, respecting YURI_DISABLE_*
#################################################################

find_package(PNG)
find_package(CURL)
find_package(ImageMagick COMPONENTS Magick++)
IF(NOT YURI_DISABLE_SAGE)
	find_package(SAGE)
ENDIF()
IF(NOT YURI_DISABLE_CUDA)
	find_package(CUDA)
ENDIF()

IF(${CORE_IEEE1394})
	pkg_check_modules(ieee1394 libiec61883)
ENDIF()

pkg_check_modules(OpenCV opencv>=2.3.0)
find_package(OPENNI2)

find_package( X11 )

find_path(VIDEOMASTER_INCLUDE_DIR VideoMasterHD_Sdi.h
          HINTS /usr/local/include/videomaster )
          

          
#################################################################
# Add optional modules based on results of the detection
#################################################################


IF(PNG_FOUND)
	add_subdirectory(pngencoder)
	add_subdirectory(pngdecoder)
ENDIF()

IF(CURL_FOUND)
	add_subdirectory(fetcher)
ENDIF()

IF(CORE_LIBAV)
	IF (${libavutil_FOUND})
		add_subdirectory(avencoder)
	ENDIF()
	IF (${libavformat_FOUND})
		add_subdirectory(tsmuxer)
		add_subdirectory(rawavfile)
		#add_subdirectory(avdemuxer)
	ENDIF()
	IF (${libswscale_FOUND})
		add_subdirectory(avscaler)
		add_subdirectory(avdecoder)
	ENDIF()
ENDIF()

IF(${CORE_CUDA})
	add_definitions("-DYURI_HAVE_CUDA")
ENDIF()

IF(${CORE_X11})
	add_subdirectory(simplerenderer)
	add_subdirectory(fb_grab)
ENDIF()

IF(${SAGE_FOUND})
	add_subdirectory(sage)
ENDIF()

IF(${CORE_ASIO})
	add_subdirectory(vncclient)
	add_subdirectory(tsrtp_receiver)
	add_subdirectory(tsrtp_streamer)
	add_subdirectory(irc_client)
ENDIF()

IF(${ImageMagick_Magick++_FOUND})
	add_subdirectory(imagemagick)
ENDIF()

IF(${OpenCV_FOUND})
	add_subdirectory(opencv)
	add_subdirectory(opencv_calib)
ENDIF()

IF(${OPENNI2_FOUND})
	add_subdirectory(openni)
ENDIF()

IF(X11_FOUND)
	add_subdirectory(screen_grab)
ENDIF()

IF (Boost_REGEX_FOUND)
	add_subdirectory(rawfilesource)
ENDIF()

IF(VIDEOMASTER_INCLUDE_DIR)
	add_subdirectory(delta)
ENDIF()

IF(UNIX)
	add_subdirectory(event_device)
ENDIF()


ENDIF()