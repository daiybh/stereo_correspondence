find_package( Boost 1.46 COMPONENTS regex)
include_directories( ${Boost_INCLUDE_DIRS})
link_directories (${Boost_LIBRARY_DIRS})



add_subdirectory(color_key)
add_subdirectory(diff)
add_subdirectory(dup)
add_subdirectory(null)
add_subdirectory(pad)
add_subdirectory(testcard)

CHECK_INCLUDE_FILES (linux/videodev2.h HAVE_VIDEODEV2_H)

IF(HAVE_VIDEODEV2_H)
	add_subdirectory(v4l2)
ENDIF()

find_package(SDL)
IF(SDL_FOUND)
	add_subdirectory(sdl_window)
ENDIF()

IF(FALSE)
#################################################################
# Add modules that are compile unconditionally
#################################################################
add_subdirectory(anaglyph)
add_subdirectory(blank)
add_subdirectory(crop)

add_subdirectory(dummy)

add_subdirectory(filedump)
add_subdirectory(flip)

add_subdirectory(split)
add_subdirectory(dxt_compress)
add_subdirectory(temperature)
add_subdirectory(read_pcap)
add_subdirectory(fps_fixer)
add_subdirectory(ts_demuxer)
add_subdirectory(combine)
add_subdirectory(split_frames)
add_subdirectory(merge_frames)
add_subdirectory(rotate)
add_subdirectory(repack_audio)

add_subdirectory(overlay)

add_subdirectory(event_info)
add_subdirectory(fade)


#################################################################
# Let's check for libraries required 
# for optional modules, respecting YURI_DISABLE_*
#################################################################
find_package(JPEG)
find_package(PNG)
find_package(CURL)
find_package(ImageMagick COMPONENTS Magick++)
IF(NOT YURI_DISABLE_SAGE)
	find_package(SAGE)
ENDIF()
IF(NOT YURI_DISABLE_CUDA)
	find_package(CUDA)
ENDIF()

IF(NOT YURI_DISABLE_LIBAV)
	pkg_check_modules(libavformat libavformat) 
	pkg_check_modules(libavutil libavutil)
	pkg_check_modules(libswscale libswscale)
ENDIF()
IF(${CORE_IEEE1394})
	pkg_check_modules(ieee1394 libiec61883)
ENDIF()

pkg_check_modules(OpenCV opencv>=2.3.0)
find_package(OPENNI2)

IF (CORE_DECKLINK)
	find_package(DECKLINK)
ENDIF()
find_package( X11 )

find_path(VIDEOMASTER_INCLUDE_DIR VideoMasterHD_Sdi.h
          HINTS /usr/local/include/videomaster )
          
find_package(ALSA)
          
#################################################################
# Add optional modules based on results of the detection
#################################################################
IF(JPEG_FOUND)
	add_subdirectory(jpegdecoder)
	add_subdirectory(jpegencoder)
ENDIF()


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
add_subdirectory(yuriconvert)

IF(${CORE_X11})
	add_subdirectory(simplerenderer)
	add_subdirectory(fb_grab)
ENDIF()

IF(${SAGE_FOUND})
	add_subdirectory(sage)
ENDIF()

IF(${CORE_IEEE1394})
	add_subdirectory(dvsource)
	add_subdirectory(hdvsource)
ENDIF()

IF(${CORE_ASIO})
	add_subdirectory(vncclient)
	add_subdirectory(tsrtp_receiver)
	add_subdirectory(tsrtp_streamer)
	add_subdirectory(osc_receiver)
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

IF(CORE_DECKLINK AND DECKLINK_FOUND)
	add_subdirectory(decklink_input)
	add_subdirectory(decklink_output)
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
	IF(ALSA_FOUND)
		add_subdirectory(midi_device)
	ENDIF()
ENDIF()


ENDIF()