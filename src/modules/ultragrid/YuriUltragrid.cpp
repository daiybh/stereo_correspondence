/*!
 * @file 		YuriUltragrid.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		16.10.2013
 * @date		21.11.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */


#include "YuriUltragrid.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/frame/CompressedVideoFrame.h"
#include "yuri/core/frame/compressed_frame_params.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include "yuri/core/utils/array_range.h"
#include <unordered_map>

extern "C" {
	#include "host.h"
}
//#include "video_frame.h"


namespace yuri {
namespace ultragrid {


namespace {

std::unordered_map<codec_t, format_t> uv_to_yuri_raw_formats =
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

std::unordered_map<codec_t, format_t> uv_to_yuri_compressed_formats =
{
	{VIDEO_CODEC_NONE, core::compressed_frame::unknown},
	{DXT1, core::compressed_frame::dxt1},
	{DXT5, core::compressed_frame::dxt5},
	{JPEG, core::compressed_frame::jpeg},
	{MJPG, core::compressed_frame::mjpg},
	{H264, core::compressed_frame::h264},
	{VP8, core::compressed_frame::vp8},
};

std::unordered_map<codec_t, std::string> uv_to_strings =
{
	{VIDEO_CODEC_NONE, "NONE"},
	{RGBA,	"RGBA"},
	{BGR, 	"BGR"},
	{RGB,	"RGB"},
	{UYVY,	"UYVY"},
	{YUYV,	"YUYV"},

	{Vuy2,	"VYU2"},
	{DVS8,	"DVS8"},

	{DXT1,	"DXT1"},
	{DXT5,	"DXT5"},
	{MJPG,	"MJPEG"},
	{H264,	"H.264"},
	{VP8,	"VP8"}
};

void video_frame_deleter(video_frame* frame)
{
	for (auto& ptr: array_range<tile>(frame->tiles, frame->tile_count)) {
		delete [] ptr.data;
		ptr.data = nullptr;
	}
	vf_free(frame);
}

}

codec_t yuri_to_uv(format_t x)
{
	for (const auto& f: uv_to_yuri_raw_formats) {
		if (f.second == x) return f.first;
	}
	return VIDEO_CODEC_NONE;
}
format_t uv_to_yuri(codec_t x)
{
	auto it = uv_to_yuri_raw_formats.find(x);
	if (it==uv_to_yuri_raw_formats.end()) return core::raw_format::unknown;
	return it->second;
}

codec_t yuri_to_uv_compressed(format_t x)
{
	for (const auto& f: uv_to_yuri_compressed_formats) {
		if (f.second == x) return f.first;
	}
	return VIDEO_CODEC_NONE;
}
format_t uv_to_yuri_compressed(codec_t x)
{
	auto it = uv_to_yuri_compressed_formats.find(x);
	if (it==uv_to_yuri_compressed_formats.end()) return core::compressed_frame::unknown;
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
	if (codec == VIDEO_CODEC_NONE) {
		codec = yuri_to_uv_compressed(fmt);
	}
	if (codec != VIDEO_CODEC_NONE) {
		return uv_to_string(codec);
	}
	return {};
}
core::pFrame copy_from_from_uv(const video_frame* uv_frame, log::Log& log)
{
	core::pFrame frame;
	const auto& tile = uv_frame->tiles[0];
	resolution_t res = {tile.width, tile.height};
	format_t fmt = ultragrid::uv_to_yuri(uv_frame->color_spec);
	if (fmt) {
		log[log::debug] << "Received frame "<<res<<" in format '"<< core::raw_format::get_format_name(fmt) << "' with " << uv_frame->tile_count << " tiles";
		/*frame = */return core::RawVideoFrame::create_empty(fmt,
					res,
					reinterpret_cast<const uint8_t*>(tile.data),
					static_cast<size_t>(tile.data_len),
					true );
	} else if ((fmt = uv_to_yuri_compressed(uv_frame->color_spec))) {
		log[log::debug] << "Received compressed frame "<<res<<" in format '"<< core::compressed_frame::get_format_name(fmt) << "' with " << uv_frame->tile_count << " tiles";
		frame = core::CompressedVideoFrame::create_empty(fmt,
					res,
					reinterpret_cast<const uint8_t*>(tile.data),
					static_cast<size_t>(tile.data_len));
	} else {
		log[log::warning] << "Unsupported frame format (" << uv_frame->color_spec <<")";
	}
	if (frame) frame->set_duration(1_s/uv_frame->fps);
	return frame;

}

bool copy_to_uv_frame(const core::pFrame& frame_in, video_frame* frame_out)
{
	if (core::pRawVideoFrame raw_frame = dynamic_pointer_cast<core::RawVideoFrame>(frame_in)) {
		return copy_to_uv_frame(raw_frame, frame_out);
	}
	if (core::pCompressedVideoFrame comp_frame = dynamic_pointer_cast<core::CompressedVideoFrame>(frame_in)) {
		return copy_to_uv_frame(comp_frame, frame_out);
	}
	return false;
}
bool copy_to_uv_frame(const core::pRawVideoFrame& frame_in, video_frame* frame_out)
{
	if (!frame_in || !frame_out) return false;
	format_t fmt = yuri_to_uv(frame_in->get_format());
	if (!fmt || (fmt != frame_out->color_spec)) return false;

	auto beg = PLANE_RAW_DATA(frame_in,0);
	// TODO OK, this is ugly and NEEDs to be redone...
	std::copy(beg, beg + frame_out->tiles[0].data_len, frame_out->tiles[0].data);
	return true;
}

bool copy_to_uv_frame(const core::pCompressedVideoFrame& frame_in, video_frame* frame_out)
{
	if (!frame_in || !frame_out) return false;
	format_t fmt = yuri_to_uv_compressed(frame_in->get_format());
	if (!fmt || (fmt != frame_out->color_spec)) return false;

	auto beg = frame_in->cbegin();
	// TODO OK, this is ugly and NEEDs to be redone...
	std::copy(beg, beg + std::min<size_t>(frame_out->tiles[0].data_len, frame_in->size()), frame_out->tiles[0].data);
	return true;
}

video_frame_t allocate_uv_frame(const core::pRawVideoFrame& in_frame)
{
	codec_t uv_fmt = yuri_to_uv(in_frame->get_format());
	if (!uv_fmt) return nullptr;

//	video_frame* out_frame = vf_alloc(1);
	video_frame_t out_frame (vf_alloc(1),video_frame_deleter);
	if (out_frame) {
		auto &tile = out_frame->tiles[0];
		tile.data=new char[PLANE_SIZE(in_frame,0)];
		tile.data_len = PLANE_SIZE(in_frame,0);
//#warning Where's the deleter now??????
		//out_frame->data_deleter = video_frame_deleter;
		resolution_t res = in_frame->get_resolution();
		tile.width = res.width;
		tile.height = res.height;
		out_frame->color_spec = uv_fmt;
		out_frame->interlacing = PROGRESSIVE;
		out_frame->fragment = 0;
		int64_t dur_val = in_frame->get_duration().value;
		out_frame->fps = dur_val?1e6/dur_val:30;
	}
	if (!copy_to_uv_frame(in_frame, out_frame)) {
		if (out_frame) {
			out_frame.reset();
//			vf_free(out_frame);
//			out_frame = nullptr;
		}
	}
	return out_frame;
}

video_frame_t allocate_uv_frame(const core::pCompressedVideoFrame& in_frame)
{
	video_frame_t out_frame (vf_alloc(1),video_frame_deleter);
	if (out_frame) {
		auto &tile = out_frame->tiles[0];
		tile.data=new char[in_frame->size()];
		tile.data_len = in_frame->size();
//#warning Where's the deleter now??????
		//out_frame->data_deleter = video_frame_deleter;
		resolution_t res = in_frame->get_resolution();
		tile.width = res.width;
		tile.height = res.height;
		out_frame->color_spec = yuri_to_uv_compressed(in_frame->get_format());
		out_frame->interlacing = PROGRESSIVE;
		out_frame->fragment = 0;
		int64_t dur_val = in_frame->get_duration().value;
		out_frame->fps = dur_val?1e6/dur_val:30;
	}
	if (!copy_to_uv_frame(in_frame, out_frame)) {
		if (out_frame) {
//			vf_free(out_frame);
//			out_frame = nullptr;
			out_frame.reset();
		}
	}
	return out_frame;
}


video_frame_t allocate_uv_frame(const core::pFrame& in_frame)
{
	if (core::pRawVideoFrame raw_frame = dynamic_pointer_cast<core::RawVideoFrame>(in_frame)) {
		return allocate_uv_frame(raw_frame);
	}
	if (core::pCompressedVideoFrame comp_frame = dynamic_pointer_cast<core::CompressedVideoFrame>(in_frame)) {
		return allocate_uv_frame(comp_frame);
	}
	return nullptr;

}

extern "C" {
void (*exit_uv)(int status);
void exit_uv_yuri(int ) {
	throw std::runtime_error("Ultragrid exit!");
}



}
void init_uv() {
	static std::mutex mutex_;
	std::unique_lock<std::mutex> _(mutex_);
	if (!exit_uv) exit_uv = exit_uv_yuri;
}

}


}



