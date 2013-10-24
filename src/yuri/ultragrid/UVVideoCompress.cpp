/*
 * UVVideoCompress.cpp
 *
 *  Created on: 24.10.2013
 *      Author: neneko
 */

#include "UVVideoCompress.h"
#include "yuri/core/Module.h"
#include "yuri/ultragrid/YuriUltragrid.h"

namespace yuri {
namespace ultragrid {
UVVideoCompress::UVVideoCompress(const log::Log &log_, core::pwThreadBase parent, const std::string& name, detail::uv_video_compress_params uv_compress_params)
:core::SpecializedIOFilter<core::RawVideoFrame>(log_,parent, name),
encoder_(nullptr), uv_compress_params_(uv_compress_params)
{

}

UVVideoCompress::~UVVideoCompress() noexcept
{

}
bool UVVideoCompress::init_compressor(const std::string& params)
{
	video_compress_params par{params.c_str()};
	if (uv_compress_params_.init_func) {
		encoder_ = uv_compress_params_.init_func(nullptr,&par);
	}
	return (encoder_ != nullptr);
}

core::pFrame UVVideoCompress::do_special_single_step(const core::pRawVideoFrame& frame)
{
	video_frame * uv_frame = ultragrid::allocate_uv_frame(frame);
	video_frame * out_uv_frame  = uv_compress_params_.compress_func(encoder_, uv_frame, 0);
	vf_free(uv_frame);

	if (out_uv_frame) {
		core::pFrame out_frame = ultragrid::copy_from_from_uv(out_uv_frame, log);
		if (out_frame) return {out_frame};
	}
	return {};
}

}
}

