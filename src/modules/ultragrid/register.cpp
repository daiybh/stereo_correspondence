/*
 * register.cpp
 *
 *  Created on: 21.10.2013
 *      Author: neneko
 */
#include "yuri/core/Module.h"
#include "yuri/core/thread/ConverterRegister.h"
#include "yuri/core/socket/DatagramSocketGenerator.h"

#ifdef YURI_UV_ALSA_SUPPORTED
#include "UVAlsaInput.h"
#include "UVAlsaOutput.h"
#endif

#include "UVAudioTestcard.h"
#include "UVConvert.h"

#ifdef YURI_UV_GLUT_SUPPORTED
#include "UVGl.h"
#endif
#ifdef YURI_UV_JPEG_SUPPORTED
#include "UVJpegCompress.h"
#endif

#ifdef YURI_UV_LIBAV_SUPPORTED
#include "UVLibav.h"
#endif

#ifdef YURI_UV_RTDXT_SUPPORTED
#include "UVRTDxt.h"
#endif

#include "UVRtpReceiver.h"
#include "UVRtpSender.h"

#if YURI_UV_SCREEN_SUPPORTED
#include "UVScreen.h"
#endif

#if YURI_UV_SDL_SUPPORTED
#include "UVSdl.h"
#endif

#include "UVTestcard.h"
#include "UVUdpSocket.h"
#include "UVUyvy.h"

#if YURI_UV_V4L2_SUPPORTED
#include "UVV4l2.h"
#endif

#if YURI_UV_DELTA_SUPPORTED
#include "UVDeltaCast.h"
#include "UVDeltaCastDVI.h"
#endif

namespace yuri {


MODULE_REGISTRATION_BEGIN("ultragrid")

#ifdef YURI_UV_ALSA_SUPPORTED
		REGISTER_IOTHREAD("uv_alsa_input",uv_alsa_input::UVAlsaInput)
		REGISTER_IOTHREAD("uv_alsa_output",uv_alsa_output::UVAlsaOutput)
#endif
		REGISTER_IOTHREAD("uv_audio_testcard",uv_audio_testcard::UVAudioTestcard)

		REGISTER_IOTHREAD("uv_convert",uv_convert::UVConvert)
		for (const auto&x: uv_convert::get_map()) {
			REGISTER_CONVERTER(x.first.first, x.first.second, "uv_convert", uv_convert::get_cost(x.first.first, x.first.second))
		}

#ifdef YURI_UV_GLUT_SUPPORTED
		REGISTER_IOTHREAD("uv_gl",uv_gl::UVGl)
#endif

#ifdef YURI_UV_JPEG_SUPPORTED
		REGISTER_IOTHREAD("uv_jpeg_compress",uv_jpeg_compress::UVJpegCompress)
#endif

#ifdef YURI_UV_LIBAV_SUPPORTED
		REGISTER_IOTHREAD("uv_libav",uv_libav::UVLibav)
#endif
#ifdef YURI_UV_RTDXT_SUPPORTED
		REGISTER_IOTHREAD("uv_rtdxt",uv_rtdxt::UVRTDxt)
#endif
		REGISTER_IOTHREAD("uv_rtp_receiver",uv_rtp_receiver::UVRtpReceiver)
		REGISTER_IOTHREAD("uv_rtp_sender",uv_rtp_sender::UVRtpSender)

#if YURI_UV_SCREEN_SUPPORTED
		REGISTER_IOTHREAD("uv_screen",uv_screen::UVScreen)
#endif

#if YURI_UV_SDL_SUPPORTED
		REGISTER_IOTHREAD("uv_sdl",uv_sdl::UVSdl)
#endif

		REGISTER_IOTHREAD("uv_testcard",uv_testcard::UVTestcard)

		REGISTER_IOTHREAD("uv_uyvy",uv_uyvy::UVUyvy)

#if YURI_UV_V4L2_SUPPORTED
		REGISTER_IOTHREAD("uv_v4l2",uv_v4l2::UVV4l2)
#endif

#if YURI_UV_DELTA_SUPPORTED
		REGISTER_IOTHREAD("uv_deltacast",uv_deltacast::UVDeltaCast)
		REGISTER_IOTHREAD("uv_deltacast_dvi",uv_deltacast::UVDeltaCastDVI)
#endif


		REGISTER_DATAGRAM_SOCKET("uv_udp",uv_udp::UVUdpSocket)

MODULE_REGISTRATION_END()


}


