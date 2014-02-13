/*!
 * @file 		UVV4l2.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		16.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "UVV4l2.h"
#include "yuri/core/Module.h"
#include "YuriUltragrid.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/RawVideoFrame.h"
extern "C" {
#include "video_capture/v4l2.h"
#include "video_capture.h"
}
namespace yuri {
namespace uv_v4l2 {


IOTHREAD_GENERATOR(UVV4l2)

core::Parameters UVV4l2::configure()
{
	core::Parameters p = ultragrid::UVVideoSource::configure();
	p.set_description("UVV4l2");
	p["path"]="/dev/video0";
	p["fps"]=30;
	p["format"]="YUYV";
	p["resolution"]=resolution_t{800,600};
	return p;
}


UVV4l2::UVV4l2(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
ultragrid::UVVideoSource(log_,parent,"uv_v4l2",UV_CAPTURE_DETAIL(v4l2))
{
	IOTHREAD_INIT(parameters)

	codec_t uv_fmt = ultragrid::yuri_to_uv(format_);
	if (uv_fmt == VIDEO_CODEC_NONE) {
		log[log::fatal] << "Specified format is not supported in ultragrid";
		throw exception::InitializationFailed("Unsupported format");
	}
	std::string uv_fmt_str = ultragrid::uv_to_string(uv_fmt);

	std::stringstream strs;
	strs << /*"v4l2:" <<*/ device_ << ":" << uv_fmt_str << ":" << resolution_.width << ":" << resolution_.height << ":1/" << fps_;// << ":"

	if (!init_capture(strs.str())) {
		throw exception::InitializationFailed("Failed to initialize v4l2 device!");
	}
}

UVV4l2::~UVV4l2() noexcept
{
}

bool UVV4l2::set_param(const core::Parameter& param)
{
	if (param.get_name() == "path") {
		device_ = param.get<std::string>();
	} else if (param.get_name() == "fps") {
		fps_ = param.get<int>();
	} else if (param.get_name() == "resolution") {
		resolution_ = param.get<resolution_t>();
	} else if (param.get_name() == "format") {
		format_ = core::raw_format::parse_format(param.get<std::string>());
	} else return ultragrid::UVVideoSource::set_param(param);
	return true;
}

} /* namespace uv_v4l2 */
} /* namespace yuri */
