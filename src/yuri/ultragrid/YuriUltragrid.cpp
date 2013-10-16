/*
 * YuriUltragrid.cpp
 *
 *  Created on: 16.10.2013
 *      Author: neneko
 */

#include "YuriUltragrid.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include <unordered_map>
//#include "video_codec.h"

namespace yuri {
namespace ultragrid {


namespace {


std::unordered_map<codec_t, format_t> uv_to_yuri_formats =
{
	{VIDEO_CODEC_NONE, core::raw_format::unknown},
	{RGBA, core::raw_format::rgba32},
	{BGR, core::raw_format::bgr24},
	{UYVY, core::raw_format::uyvy422},
	{YUYV, core::raw_format::yuyv422},

	{Vuy2, core::raw_format::uyvy422},
	{DVS8, core::raw_format::uyvy422},

};

std::unordered_map<codec_t, std::string> uv_to_strings =
{
	{VIDEO_CODEC_NONE, "NONE"},
	{RGBA, "RGBA"},
	{BGR, "BGR"},
	{UYVY, "UYVY"},
	{YUYV, "YUYV"},

	{Vuy2, "VYU2"},
	{DVS8, "DVS8"},
};

}

codec_t yuri_to_uv(format_t x)
{
	for (const auto& f: uv_to_yuri_formats) {
		if (f.second == x) return f.first;
	}
	return VIDEO_CODEC_NONE;
}
format_t uv_to_yuri(codec_t x)
{
	auto it = uv_to_yuri_formats.find(x);
	if (it==uv_to_yuri_formats.end()) return core::raw_format::unknown;
	return it->second;
}
std::string uv_to_string(codec_t x)
{
	auto it = uv_to_strings.find(x);
	if (it==uv_to_strings.end()) return "";
	return it->second;
}

core::pFrame copy_from_from_uv(const video_frame* uv_frame, log::Log& log)
{
	core::pFrame frame;
	format_t fmt = ultragrid::uv_to_yuri(uv_frame->color_spec);
	if (!fmt) {
		return frame;
	}


	const auto& tile = uv_frame->tiles[0];
	resolution_t res = {tile.width, tile.height};
	log[log::info] << "Received frame "<<res<<" in format '"<< core::raw_format::get_format_name(fmt) << "' with " << uv_frame->tile_count << " tiles";

	frame = core::RawVideoFrame::create_empty(fmt,
				res,
				reinterpret_cast<const uint8_t*>(tile.data),
				static_cast<size_t>(tile.data_len),
				true );
	return frame;

}

}
}


extern "C" {
void exit_uv(int ) {
	throw std::runtime_error("Ultragrid exit!");
}

}



