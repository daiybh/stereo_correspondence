/*
 * libav.cpp
 *
 *  Created on: 29.10.2013
 *      Author: neneko
 */
#include "libav.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include <unordered_map>
namespace yuri {
namespace libav {

namespace {
using namespace yuri::core::compressed_frame;
std::unordered_map<format_t, CodecID> yuri_codec_map = {
		{mpeg2, 					CODEC_ID_MPEG2VIDEO},
		{mpeg1, 					CODEC_ID_MPEG1VIDEO},
//		{YURI_VIDEO_HUFFYUV, 		CODEC_ID_HUFFYUV},
		{dv, 						CODEC_ID_DVVIDEO},
		{mpeg2ts,					CODEC_ID_MPEG2TS},
		{mjpg,						CODEC_ID_MJPEG},
		{h264,						CODEC_ID_H264},
//		{YURI_VIDEO_FLASH,			CODEC_ID_FLASHSV2},
//		{YURI_VIDEO_DIRAC,			CODEC_ID_DIRAC},
//		{YURI_VIDEO_H263,			CODEC_ID_H263},
//		{YURI_VIDEO_H263PLUS,		CODEC_ID_H263P},
//		{YURI_VIDEO_THEORA,			CODEC_ID_THEORA},
		{vp8,						CODEC_ID_VP8},
};
}
namespace {
using namespace yuri::core::raw_format;
std::unordered_map<format_t, PixelFormat> yuri_pixel_map = {
		{rgb8,			 			PIX_FMT_RGB8},
		{bgr8,			 			PIX_FMT_BGR8},
		{rgb24,			 			PIX_FMT_RGB24},
		{bgr24,			 			PIX_FMT_BGR24},
		{rgba32,		 			PIX_FMT_RGBA},
		{yuyv422,					PIX_FMT_YUYV422},
		{yuv420p,					PIX_FMT_YUV420P},
		{yuv422p,					PIX_FMT_YUV422P},
		{yuv444p,					PIX_FMT_YUV444P},
		{uyvy422,					PIX_FMT_UYVY422},
		{yuv411p,					PIX_FMT_YUV411P},
};

}


PixelFormat avpixelformat_from_yuri(yuri::format_t format)
{
	auto it = yuri_pixel_map.find(format);
	return (it==yuri_pixel_map.end())?PIX_FMT_NONE:it->second;
}
CodecID avcodec_from_yuri_format(yuri::format_t codec)
{
	auto it = yuri_codec_map.find(codec);
	return (it==yuri_codec_map.end())?CODEC_ID_NONE:it->second;
}


yuri::format_t yuri_pixelformat_from_av(PixelFormat format)
{
	for (auto f: yuri_pixel_map) {
		if (f.second == format) return f.first;
	}
	return 0;
}
yuri::format_t yuri_format_from_avcodec(CodecID codec)
{
	for (auto f: yuri_codec_map) {
		if (f.second == codec) return f.first;
	}
	return 0;
}
}
}



