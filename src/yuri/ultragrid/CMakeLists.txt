
SET(YURI_HELPER_UV_SRC ${YURI_HELPER_DECKLINK_SRC}
	ultragrid/YuriUltragrid.cpp ultragrid/YuriUltragrid.h
	${YURI_HELPER_UV_INCLUDE}/capture_filter.c
	${YURI_HELPER_UV_INCLUDE}/debug.c
	${YURI_HELPER_UV_INCLUDE}/host.c
	${YURI_HELPER_UV_INCLUDE}/lib_common.c
	${YURI_HELPER_UV_INCLUDE}/messaging.cpp
	${YURI_HELPER_UV_INCLUDE}/module.c
	${YURI_HELPER_UV_INCLUDE}/tv.c
	${YURI_HELPER_UV_INCLUDE}/video_capture.c
	${YURI_HELPER_UV_INCLUDE}/video_codec.c
	${YURI_HELPER_UV_INCLUDE}/video_display.c
	${YURI_HELPER_UV_INCLUDE}/video_frame.c
	${YURI_HELPER_UV_INCLUDE}/x11_common.c
	
	${YURI_HELPER_UV_INCLUDE}/audio/codec.c
	${YURI_HELPER_UV_INCLUDE}/audio/utils.c
	${YURI_HELPER_UV_INCLUDE}/audio/wav_reader.c
	${YURI_HELPER_UV_INCLUDE}/audio/codec/dummy_pcm.c
	
	
	${YURI_HELPER_UV_INCLUDE}/capture_filter/blank.c
	${YURI_HELPER_UV_INCLUDE}/capture_filter/none.c
	${YURI_HELPER_UV_INCLUDE}/capture_filter/every.c
	${YURI_HELPER_UV_INCLUDE}/capture_filter/logo.c
	
	${YURI_HELPER_UV_INCLUDE}/compat/platform_spin.c
	
	${YURI_HELPER_UV_INCLUDE}/crypto/random.c
	
	${YURI_HELPER_UV_INCLUDE}/utils/config_file.c
	${YURI_HELPER_UV_INCLUDE}/utils/list.c
	${YURI_HELPER_UV_INCLUDE}/utils/resource_manager.cpp
	${YURI_HELPER_UV_INCLUDE}/utils/ring_buffer.c
	
	
	#${YURI_HELPER_UV_INCLUDE}/video_capture/aggregate.c
	#${YURI_HELPER_UV_INCLUDE}/video_capture/import.c
	${YURI_HELPER_UV_INCLUDE}/video_capture/null.c
	#${YURI_HELPER_UV_INCLUDE}/video_capture/rtsp.c
	
	
	#${YURI_HELPER_UV_INCLUDE}/video_display/aggregate.c
	#${YURI_HELPER_UV_INCLUDE}/video_capture/import.c
	${YURI_HELPER_UV_INCLUDE}/video_display/null.c
	
	
	
	PARENT_SCOPE)
