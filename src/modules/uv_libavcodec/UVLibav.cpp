/*!
 * @file 		UVLibav.cpp
 * @author 		<Your name>
 * @date		17.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "UVLibav.h"
#include "yuri/core/Module.h"
#include "yuri/ultragrid/YuriUltragrid.h"
#include "yuri/core/frame/compressed_frame_params.h"
#include "yuri/core/frame/compressed_frame_types.h"

extern "C" {
#include "video_compress.h"
#include "video_compress/libavcodec.h"
#include "video_frame.h"

}

namespace yuri {
namespace uv_rtdxt {



IOTHREAD_GENERATOR(UVLibav)

MODULE_REGISTRATION_BEGIN("uv_libav")
		REGISTER_IOTHREAD("uv_libav",UVLibav)
MODULE_REGISTRATION_END()

core::Parameters UVLibav::configure()
{
	core::Parameters p = ultragrid::UVVideoCompress::configure();
	p.set_description("UVLibav");
	p["format"]["Output format for encoding"]="MJPG";
	p["bps"]["Requested bit rate"]=-1;
	p["subsampling"]["Requested subsampling. Supported values 420 and 422"]="422";
	p["preset"]["Preset for H.264 encoder"]="";
	return p;
}


UVLibav::UVLibav(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
ultragrid::UVVideoCompress(log_,parent, "uv_libav", UV_COMPRESS_DETAIL_TILE(libavcodec))
{
	IOTHREAD_INIT(parameters)

	std::stringstream sstr;
	std::string uv_fmt = ultragrid::yuri_to_uv_string(format_);
	if (uv_fmt.empty()) {
		log[log::fatal] << "Unsupported codec!";
		throw exception::InitializationFailed("Unsupported codec");
	}
	sstr << "codec=" << uv_fmt;
	if (bps_ > 0) sstr << ":bitrate="<<bps_;
	if (!subsampling_.empty()) sstr << ":subsampling="<<subsampling_;
	if (!preset_.empty()) sstr << ":preset="<<preset_;

	if(!init_compressor(sstr.str())) {
		log[log::fatal] << "Failed to create encoder";
		throw exception::InitializationFailed("Failed to create libavcodec encoder");
	}
}

UVLibav::~UVLibav() noexcept
{
}


bool UVLibav::set_param(const core::Parameter& param)
{
	if (param.get_name() == "format") {
		format_ = core::compressed_frame::parse_format(param.get<std::string>());
		if (!format_) format_ = core::compressed_frame::mjpg;
	} else if (param.get_name() == "bps") {
		bps_ = param.get<ssize_t>();
	} else if (param.get_name() == "subsampling") {
		subsampling_ = param.get<std::string>();
	} else if (param.get_name() == "preset") {
		preset_ = param.get<std::string>();
	} else return ultragrid::UVVideoCompress::set_param(param);
	return true;
}

} /* namespace uv_rtdxt */
} /* namespace yuri */
