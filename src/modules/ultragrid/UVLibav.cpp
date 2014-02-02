/*!
 * @file 		UVLibav.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		17.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "UVLibav.h"
#include "yuri/core/Module.h"
#include "yuri/ultragrid/YuriUltragrid.h"
#include "yuri/core/frame/compressed_frame_params.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include "yuri/core/thread/ConverterRegister.h"

extern "C" {
#include "video_compress.h"
#include "video_compress/libavcodec.h"
#include "video_frame.h"

}

namespace yuri {
namespace uv_libav {



IOTHREAD_GENERATOR(UVLibav)

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
