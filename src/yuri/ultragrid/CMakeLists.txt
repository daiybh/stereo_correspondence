
SET(YURI_HELPER_UV_SRC ${YURI_HELPER_UV_SRC}
	ultragrid/YuriUltragrid.cpp 	ultragrid/YuriUltragrid.h
	ultragrid/UVVideoSink.cpp 		ultragrid/UVVideoSink.h
	ultragrid/UVVideoSource.cpp 	ultragrid/UVVideoSource.h
	ultragrid/UVVideoCompress.cpp 	ultragrid/UVVideoCompress.h
	
	${YURI_HELPER_UV_INCLUDE}/capture_filter.c
	${YURI_HELPER_UV_INCLUDE}/debug.c
	${YURI_HELPER_UV_INCLUDE}/host.c
	${YURI_HELPER_UV_INCLUDE}/lib_common.c
	${YURI_HELPER_UV_INCLUDE}/messaging.cpp
	${YURI_HELPER_UV_INCLUDE}/module.c
	${YURI_HELPER_UV_INCLUDE}/ntp.c
	${YURI_HELPER_UV_INCLUDE}/pdb.c
	${YURI_HELPER_UV_INCLUDE}/tfrc.c
	${YURI_HELPER_UV_INCLUDE}/transmit.c
	${YURI_HELPER_UV_INCLUDE}/tv.c
	${YURI_HELPER_UV_INCLUDE}/vf_split.c
	${YURI_HELPER_UV_INCLUDE}/video_capture.c
	${YURI_HELPER_UV_INCLUDE}/video_codec.c
	${YURI_HELPER_UV_INCLUDE}/video_compress.cpp
	${YURI_HELPER_UV_INCLUDE}/video_display.c
	${YURI_HELPER_UV_INCLUDE}/video_frame.c
	${YURI_HELPER_UV_INCLUDE}/x11_common.c
	
	${YURI_HELPER_UV_INCLUDE}/audio/audio.c
	${YURI_HELPER_UV_INCLUDE}/audio/audio_capture.c
	${YURI_HELPER_UV_INCLUDE}/audio/audio_playback.c
	${YURI_HELPER_UV_INCLUDE}/audio/codec.c
	${YURI_HELPER_UV_INCLUDE}/audio/echo.c
	${YURI_HELPER_UV_INCLUDE}/audio/export.c
	${YURI_HELPER_UV_INCLUDE}/audio/resampler.c
	${YURI_HELPER_UV_INCLUDE}/audio/utils.c
	${YURI_HELPER_UV_INCLUDE}/audio/wav_reader.c
	
	${YURI_HELPER_UV_INCLUDE}/audio/capture/none.c
	${YURI_HELPER_UV_INCLUDE}/audio/capture/sdi.c
	
	${YURI_HELPER_UV_INCLUDE}/audio/codec/dummy_pcm.c
	
	${YURI_HELPER_UV_INCLUDE}/audio/playback/none.c
	${YURI_HELPER_UV_INCLUDE}/audio/playback/sdi.c
	
	
	${YURI_HELPER_UV_INCLUDE}/capture_filter/blank.c
	${YURI_HELPER_UV_INCLUDE}/capture_filter/none.c
	${YURI_HELPER_UV_INCLUDE}/capture_filter/every.c
	${YURI_HELPER_UV_INCLUDE}/capture_filter/logo.c
	
	${YURI_HELPER_UV_INCLUDE}/compat/platform_spin.c
	
	${YURI_HELPER_UV_INCLUDE}/crypto/md5.c
	${YURI_HELPER_UV_INCLUDE}/crypto/crc_32.c
	${YURI_HELPER_UV_INCLUDE}/crypto/crypt_aes.c
	${YURI_HELPER_UV_INCLUDE}/crypto/crypt_aes_impl.c
	${YURI_HELPER_UV_INCLUDE}/crypto/crypt_des.c
	${YURI_HELPER_UV_INCLUDE}/crypto/random.c
	

	${YURI_HELPER_UV_INCLUDE}/rtp/audio_decoders.c
	${YURI_HELPER_UV_INCLUDE}/rtp/ldgm.cpp
	${YURI_HELPER_UV_INCLUDE}/rtp/net_udp.c
	${YURI_HELPER_UV_INCLUDE}/rtp/pbuf.c
	${YURI_HELPER_UV_INCLUDE}/rtp/pc.cpp
	${YURI_HELPER_UV_INCLUDE}/rtp/rtp.c
	${YURI_HELPER_UV_INCLUDE}/rtp/rtp_callback.c
	
	${YURI_HELPER_UV_INCLUDE}/utils/config_file.c
	${YURI_HELPER_UV_INCLUDE}/utils/list.c
	${YURI_HELPER_UV_INCLUDE}/utils/message_queue.cpp
	${YURI_HELPER_UV_INCLUDE}/utils/resource_manager.cpp
	${YURI_HELPER_UV_INCLUDE}/utils/packet_counter.cpp
	${YURI_HELPER_UV_INCLUDE}/utils/ring_buffer.c
	${YURI_HELPER_UV_INCLUDE}/utils/worker.cpp
	
	${YURI_HELPER_UV_INCLUDE}/video_capture/null.c
	
	${YURI_HELPER_UV_INCLUDE}/video_compress/none.c
	
	${YURI_HELPER_UV_INCLUDE}/video_display/null.c
		
	${YURI_HELPER_UV_INCLUDE}/../ldgm-coding/ldgm-session.cpp
	${YURI_HELPER_UV_INCLUDE}/../ldgm-coding/ldgm-session-cpu.cpp
	${YURI_HELPER_UV_INCLUDE}/../ldgm-coding/tanner.cpp
	${YURI_HELPER_UV_INCLUDE}/../ldgm-coding/matrix-gen/ldpc-matrix.cpp
	${YURI_HELPER_UV_INCLUDE}/../ldgm-coding/matrix-gen/matrix-generator.cpp
		
	)

IF(X11_FOUND)
SET(YURI_HELPER_UV_SRC ${YURI_HELPER_UV_SRC}
	${YURI_HELPER_UV_INCLUDE}/x11_common.c
	)
ENDIF()

IF(OPENSSL_FOUND)
SET(YURI_HELPER_UV_SRC ${YURI_HELPER_UV_SRC}
	${YURI_HELPER_UV_INCLUDE}/crypto/openssl_encrypt.c
	${YURI_HELPER_UV_INCLUDE}/crypto/openssl_decrypt.c
	)
ENDIF()

SET(YURI_HELPER_UV_SRC ${YURI_HELPER_UV_SRC}
	PARENT_SCOPE)