/*
 * YuriUltragrid.cpp
 *
 *  Created on: 16.10.2013
 *      Author: neneko
 */

#include "YuriUltragrid.h"
#include <unordered_map>
#include "video_frame.h"

namespace yuri {
namespace ultragrid {


namespace {


std::unordered_map<codec_t, format_t> uv_to_yuri_formats =
{
	{VIDEO_CODEC_NONE, core::raw_format::unknown},
	{RGBA, core::raw_format::rgba32},
	{BGR, core::raw_format::bgr24},
	{RGB, core::raw_format::rgb24},
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
	{RGB, "RGB"},
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
std::string yuri_to_uv_string(format_t fmt)
{
	codec_t codec = yuri_to_uv(fmt);
	if (codec != VIDEO_CODEC_NONE) {
		return uv_to_string(codec);
	}
	return {};
}
core::pFrame copy_from_from_uv(const video_frame* uv_frame, log::Log& log)
{
	core::pFrame frame;
	format_t fmt = ultragrid::uv_to_yuri(uv_frame->color_spec);
	if (!fmt) {
		log[log::warning] << "Unsupported frame format (" << uv_frame->color_spec <<")";
		return frame;
	}


	const auto& tile = uv_frame->tiles[0];
	resolution_t res = {tile.width, tile.height};
	log[log::debug] << "Received frame "<<res<<" in format '"<< core::raw_format::get_format_name(fmt) << "' with " << uv_frame->tile_count << " tiles";

	frame = core::RawVideoFrame::create_empty(fmt,
				res,
				reinterpret_cast<const uint8_t*>(tile.data),
				static_cast<size_t>(tile.data_len),
				true );
	return frame;

}

bool copy_to_uv_frame(const core::pRawVideoFrame& frame_in, video_frame* frame_out)
{
	if (!frame_in || !frame_out) return false;
	format_t fmt = yuri_to_uv(frame_in->get_format());
	if (fmt != frame_out->color_spec) return false;
//	const auto& fi = core::raw_format::get_format_info(fmt);
	auto beg = PLANE_RAW_DATA(frame_in,0);
	// TODO OK, this is ugly and NEED to be redone...
	std::copy(beg, beg + frame_out->tiles[0].data_len, frame_out->tiles[0].data);
	return true;
}

video_frame* allocate_uv_frame(const core::pRawVideoFrame& in_frame)
{
	video_frame* out_frame = vf_alloc(1);
	if (out_frame) {
		auto &tile = out_frame->tiles[0];
		tile.data=new char[PLANE_SIZE(in_frame,0)];
		tile.data_len = PLANE_SIZE(in_frame,0);
		resolution_t res = in_frame->get_resolution();
		tile.width = res.width;
		tile.height = res.height;
		out_frame->color_spec = yuri_to_uv(in_frame->get_format());
		out_frame->interlacing = PROGRESSIVE;
		out_frame->fragment = 0;

	}

	if (!copy_to_uv_frame(in_frame, out_frame)) {
		if (out_frame) {
			if (out_frame->tiles[0].data) {
				delete [] out_frame->tiles[0].data;
				out_frame->tiles[0].data=nullptr;
			}
			vf_free(out_frame);
			out_frame = nullptr;
		}
	}
	return out_frame;

}
}
}


extern "C" {
void exit_uv(int ) {
	throw std::runtime_error("Ultragrid exit!");
}

}



