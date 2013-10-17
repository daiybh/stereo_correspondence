# Set name of the module
SET (MODULE uv_rtp_sender)

# Set all source files module uses
SET (SRC UVRtpSender.cpp
		 UVRtpSender.h
		 ${UV_BASE}/transmit.c
		 ${UV_BASE}/ntp.c
		 ${UV_BASE}/rtp/ldgm.cpp
		 ${UV_BASE}/rtp/rtp.c
		 ${UV_BASE}/rtp/pc.cpp
		 ${UV_BASE}/rtp/net_udp.c
		 ${UV_BASE}/../ldgm-coding/ldgm-session.cpp
		 ${UV_BASE}/../ldgm-coding/ldgm-session-cpu.cpp
		 #${UV_BASE}/../ldgm-coding/ldgm-session-gpu.cpp
		 ${UV_BASE}/../ldgm-coding/tanner.cpp
		 ${UV_BASE}/../ldgm-coding/matrix-gen/ldpc-matrix.cpp
		 ${UV_BASE}/../ldgm-coding/matrix-gen/matrix-generator.cpp
		 ${UV_BASE}/crypto/openssl_encrypt.c
		 ${UV_BASE}/crypto/md5.c
		 ${UV_BASE}/crypto/crc_32.c
		 ${UV_BASE}/crypto/crypt_aes.c
		 ${UV_BASE}/crypto/crypt_aes_impl.c
		 ${UV_BASE}/crypto/crypt_des.c)


include_directories(${UV_BASE} ${UV_BASE}/..) 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${LIBNAME_HELPER_UV} ${OPENSSL_LIBRARIES})
SET_TARGET_PROPERTIES(${MODULE} PROPERTIES COMPILE_FLAGS -DHAVE_CONFIG_H)


YURI_INSTALL_MODULE(${MODULE})
