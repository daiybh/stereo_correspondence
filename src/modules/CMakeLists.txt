find_package( Boost 1.46 COMPONENTS regex)
include_directories( ${Boost_INCLUDE_DIRS})
link_directories (${Boost_LIBRARY_DIRS})


add_subdirectory(anaglyph)
add_subdirectory(color_key)
add_subdirectory(color_picker)
add_subdirectory(combine)
add_subdirectory(convert_planes)
add_subdirectory(crop)
add_subdirectory(diff)
add_subdirectory(dup)
add_subdirectory(event_info)
add_subdirectory(fade)
add_subdirectory(filedump)
add_subdirectory(flip)
add_subdirectory(fps_fixer)
add_subdirectory(frame_info)
add_subdirectory(invert)
add_subdirectory(irc_client)
add_subdirectory(merge_frames)
add_subdirectory(mix)
add_subdirectory(mjpeg_to_jpeg)
add_subdirectory(mosaic)
add_subdirectory(null)
add_subdirectory(overlay)
add_subdirectory(pad)
add_subdirectory(rotate)
add_subdirectory(select)
add_subdirectory(split_frames)
add_subdirectory(testcard)
add_subdirectory(yuriconvert)
add_subdirectory(osc_receiver)
add_subdirectory(vncclient)

IF(UNIX)
	add_subdirectory(yuri_tcp)
ENDIF()

CHECK_INCLUDE_FILE_CXX (linux/videodev2.h HAVE_VIDEODEV2_H)
CHECK_INCLUDE_FILE_CXX (linux/dvb/video.h HAVE_DVB_VIDEO_H)

IF(NOT YURI_DISABLE_LIBAV)
	pkg_check_modules(libavformat libavformat) 
	pkg_check_modules(libavutil libavutil)
	pkg_check_modules(libswscale libswscale)
	pkg_check_modules(libavcodec libavcodec)
ENDIF()

IF (CORE_DECKLINK)
	find_package(DECKLINK)
ENDIF()


IF(HAVE_VIDEODEV2_H)
	add_subdirectory(v4l2)
ENDIF()

IF(HAVE_DVB_VIDEO_H)
	add_subdirectory(dvb)
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
#	add_subdirectory(jpegdecoder)
#	add_subdirectory(jpegencoder)
	add_subdirectory(jpeg)
ENDIF()


IF(NOT YURI_DISABLE_DECKLINK)
	find_package(DECKLINK)
ENDIF()

IF(DECKLINK_FOUND)
	add_subdirectory(decklink)
ENDIF()

IF(NOT YURI_DISABLE_IEEE1394)
	pkg_check_modules(ieee1394 libiec61883 libavc1394)
ENDIF()

IF(ieee1394_FOUND)
    add_subdirectory(ieee1394)
ENDIF()



IF(CORE_LIBAV AND libavcodec_FOUND AND libavformat_FOUND)
	add_subdirectory(rawavfile)
ENDIF()

find_package(PNG)
IF(PNG_FOUND)
	add_subdirectory(png)
ENDIF()

find_package(X11)
IF(X11_FOUND)
	add_subdirectory(screen_grab)
ENDIF()

find_package(ImageMagick COMPONENTS Magick++)
IF(${ImageMagick_Magick++_FOUND})
	add_subdirectory(imagemagick)
ENDIF()

pkg_check_modules(OpenCV opencv>=2.3.0)
IF(${OpenCV_FOUND})
	add_subdirectory(opencv)
	#add_subdirectory(opencv_calib)
ENDIF()

IF(NOT YURI_DISABLE_SAGE)
	find_package(SAGE)
ENDIF()
IF(${SAGE_FOUND})
	add_subdirectory(sage)
ENDIF()

pkg_check_modules(JACK jack)
IF(${JACK_FOUND})
    add_subdirectory(jack)
ENDIF()

pkg_check_modules(THEORAENC theoraenc)
IF(${THEORAENC_FOUND})
    add_subdirectory(theora)
ENDIF()


IF (Boost_REGEX_FOUND)
	add_subdirectory(rawfilesource)
ENDIF()



#### UV #####
#IF(NOT YURI_DISABLE_ULTRAGRID)
	# This should be replaced by a proper detection!
	
IF(NOT YURI_DISABLE_ULTRAGRID)
	SET(UV_BASE ${CMAKE_SOURCE_DIR}/../ultragrid/src)
	add_subdirectory(ultragrid)	
ENDIF()
#############


IF(FALSE)
#################################################################
# Add modules that are compile unconditionally
#################################################################
add_subdirectory(blank)

add_subdirectory(dummy)

add_subdirectory(split)
add_subdirectory(dxt_compress)
add_subdirectory(temperature)
add_subdirectory(read_pcap)

add_subdirectory(ts_demuxer)


add_subdirectory(repack_audio)

#################################################################
# Let's check for libraries required 
# for optional modules, respecting YURI_DISABLE_*
#################################################################

find_package(CURL)


IF(NOT YURI_DISABLE_CUDA)
	find_package(CUDA)
ENDIF()

IF(${CORE_IEEE1394})
	pkg_check_modules(ieee1394 libiec61883)
ENDIF()


find_package(OPENNI2)

find_package( X11 )

find_path(VIDEOMASTER_INCLUDE_DIR VideoMasterHD_Sdi.h
          HINTS /usr/local/include/videomaster )
          

          
#################################################################
# Add optional modules based on results of the detection
#################################################################


IF(CURL_FOUND)
	add_subdirectory(fetcher)
ENDIF()

IF(CORE_LIBAV)
	IF (${libavutil_FOUND})
		add_subdirectory(avencoder)
	ENDIF()
	IF (${libavformat_FOUND})
		add_subdirectory(tsmuxer)
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


IF(${CORE_ASIO})
	add_subdirectory(tsrtp_receiver)
	add_subdirectory(tsrtp_streamer)
ENDIF()


IF(${OPENNI2_FOUND})
	add_subdirectory(openni)
ENDIF()



IF(VIDEOMASTER_INCLUDE_DIR)
	add_subdirectory(delta)
ENDIF()

IF(UNIX)
	add_subdirectory(event_device)
ENDIF()


ENDIF()