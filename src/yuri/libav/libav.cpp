/*!
 * @file 		libav.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		29.10.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#include "libav.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include <unordered_map>
#include <map>
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
		{rgb16,						PIX_FMT_RGB555},
		{rgb16,						PIX_FMT_RGB565},
		{bgr15,						PIX_FMT_BGR555},
		{bgr16,						PIX_FMT_BGR565},
		{rgb24,			 			PIX_FMT_RGB24},
		{bgr24,			 			PIX_FMT_BGR24},
		{rgba32,		 			PIX_FMT_RGBA},
		{argb32,		 			PIX_FMT_ARGB},
		{bgra32,		 			PIX_FMT_BGRA},
		{abgr32,		 			PIX_FMT_ABGR},
		{rgb48,						PIX_FMT_RGB48},
		{bgr48,						PIX_FMT_BGR48},
		{rgba64,					PIX_FMT_RGBA64},
		{bgra64,					PIX_FMT_BGRA64},


		{y8,						PIX_FMT_GRAY8},
		{y16,						PIX_FMT_GRAY16},
		{yuv420p,					PIX_FMT_YUV420P},
		{yuv422p,					PIX_FMT_YUV422P},

		{yuv444p,					PIX_FMT_YUV444P},
		{yuyv422,					PIX_FMT_YUYV422},
		{uyvy422,					PIX_FMT_UYVY422},

		{yuv411p,					PIX_FMT_YUV411P},
};

std::map<PixelFormat, PixelFormat> yuri_pixel_special_map = {
{PIX_FMT_YUVJ420P, PIX_FMT_YUV420P},
{PIX_FMT_YUVJ422P, PIX_FMT_YUV422P},
{PIX_FMT_YUVJ444P, PIX_FMT_YUV444P},
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
	auto it = yuri_pixel_special_map.find(format);
	if (it != yuri_pixel_special_map.end()) format = it->second;

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



