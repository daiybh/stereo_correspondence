/*
 * UVVideoDecompress.cpp
 *
 *  Created on: 10.10.2014
 *      Author: neneko
 */

#include "UVVideoDecompress.h"
#include "yuri/core/Module.h"
#include "YuriUltragrid.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "video_frame.h"
#include "YuriUltragrid.h"

namespace yuri {
namespace ultragrid {


core::Parameters UVVideoDecompress::configure()
{
	core::Parameters p = base_type::configure();
	p.set_description("UV Video Decompress");
	p["format"]["Output format for decoding"]="YUV";
	return p;
}

UVVideoDecompress::UVVideoDecompress(const log::Log &log_, core::pwThreadBase parent, const std::string& name, detail::uv_video_decompress_params uv_decompress_params)
:base_type(log_,parent, name),
decoder_(nullptr), uv_decompress_params_(uv_decompress_params),last_resolution_{0,0},
last_format_(0),output_format_(core::raw_format::yuyv422),sequence_(0)
{

}

UVVideoDecompress::~UVVideoDecompress() noexcept
{

}
bool UVVideoDecompress::init_decompressor(const std::string& /* params */)
{
	if (uv_decompress_params_.init_func) {
		decoder_ = uv_decompress_params_.init_func();
	}
	return (decoder_ != nullptr);
}

core::pFrame UVVideoDecompress::do_special_single_step(core::pCompressedVideoFrame frame)
{
	const auto res = frame->get_resolution();
	const auto format = frame->get_format();
	if (last_resolution_ != res || last_format_ != format) {
		auto codec = ultragrid::yuri_to_uv_compressed(format);
		auto codec_out = ultragrid::yuri_to_uv(output_format_);
		// Ugly hack, but it seems the formats are wrong otherwise...
		if (codec_out == YUYV) codec_out=UYVY;
		video_desc vd = {static_cast<unsigned>(res.width), static_cast<unsigned>(res.height), codec, 25, PROGRESSIVE, 1};
		uv_decompress_params_.decompress_reconfigure(decoder_, vd, 0, 8, 16, 24, codec_out);
		last_resolution_ = res;
		last_format_ = format;
		sequence_ = 0;
	}

	auto out_frame = core::RawVideoFrame::create_empty(output_format_, frame->get_resolution());
	if (uv_decompress_params_.decompress_func) {
		if (uv_decompress_params_.decompress_func(decoder_, PLANE_RAW_DATA(out_frame,0), &(*frame)[0],frame->get_size(),sequence_)) {
			return out_frame;
		}
	}
	return {};
}
core::pFrame UVVideoDecompress::do_convert_frame(core::pFrame input_frame, format_t target_format)
{
	auto frame = std::dynamic_pointer_cast<core::CompressedVideoFrame>(input_frame);
	if (frame) {
		if (output_format_ != target_format) {
			output_format_ = target_format;
			last_resolution_ = {0, 0}; // Make sure to reconfigure next time
		}
		return do_special_single_step(frame);
	}
	return {};
}


bool UVVideoDecompress::set_param(const core::Parameter& param)
{
	if (param.get_name() == "format") {
		output_format_ = core::raw_format::parse_format(param.get<std::string>());
		if (!output_format_) output_format_ = core::raw_format::yuyv422;
	} return base_type::set_param(param);
	return true;
}




}
}

