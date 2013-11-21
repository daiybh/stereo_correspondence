/*!
 * @file 		UVRTDxt.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		17.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "UVRTDxt.h"
#include "yuri/core/Module.h"
#include "yuri/ultragrid/YuriUltragrid.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include "yuri/core/frame/compressed_frame_params.h"
extern "C" {
#include "video_compress.h"
#include "video_compress/dxt_glsl.h"
#include "video_frame.h"
}
namespace yuri {
namespace uv_rtdxt {


IOTHREAD_GENERATOR(UVRTDxt)

MODULE_REGISTRATION_BEGIN("uv_rtdxt")
		REGISTER_IOTHREAD("uv_rtdxt",UVRTDxt)
MODULE_REGISTRATION_END()

core::Parameters UVRTDxt::configure()
{
	core::Parameters p = ultragrid::UVVideoCompress::configure();
	p.set_description("UVRTDxt");
	p["format"]["Output format (should be DXT1 or DXT5)"]="DXT1";
	return p;
}


UVRTDxt::UVRTDxt(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
ultragrid::UVVideoCompress(log_,parent, "uv_rtdxt", UV_COMPRESS_DETAIL(dxt_glsl)),
format_(core::compressed_frame::dxt1)
{
	IOTHREAD_INIT(parameters)
	std::string uv_fmt = ultragrid::yuri_to_uv_string(format_);
	if (uv_fmt.empty()) {
		log[log::fatal] << "Wrong format";
		throw exception::InitializationFailed("Wrong output format for DXT encoder");
	}
	if(!init_compressor(uv_fmt)) {
		log[log::fatal] << "Failed to create encoder";
		throw exception::InitializationFailed("Failed to create DXT encoder");
	}
}

UVRTDxt::~UVRTDxt() noexcept
{
}


bool UVRTDxt::set_param(const core::Parameter& param)
{
	if (param.get_name() == "format") {
		using namespace core::compressed_frame;
		format_ = parse_format(param.get<std::string>());
		if (format_ != dxt1 && format_ != dxt5) {
			log[log::warning] << "Unsupported output codec, defaulting to DXT1";
			format_ = dxt1;
		}
	} else return ultragrid::UVVideoCompress::set_param(param);
	return true;
}

} /* namespace uv_rtdxt */
} /* namespace yuri */
