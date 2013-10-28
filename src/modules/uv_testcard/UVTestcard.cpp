/*!
 * @file 		UVTestcard.cpp
 * @author 		<Your name>
 * @date		16.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "UVTestcard.h"
#include "yuri/core/Module.h"
#include "yuri/ultragrid/YuriUltragrid.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/raw_frame_params.h"
extern "C" {
#include "video_capture/testcard.h"
}
namespace yuri {
namespace uv_v4l2 {


IOTHREAD_GENERATOR(UVTestcard)

MODULE_REGISTRATION_BEGIN("uv_testcard")
		REGISTER_IOTHREAD("uv_testcard",UVTestcard)
MODULE_REGISTRATION_END()

core::Parameters UVTestcard::configure()
{
	core::Parameters p = ultragrid::UVVideoSource::configure();
	p.set_description("UVTestcard");
	p["fps"]=30;
	return p;
}


UVTestcard::UVTestcard(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
		ultragrid::UVVideoSource(log_, parent, "uv_testcard", UV_CAPTURE_DETAIL(testcard)),
resolution_{800,600},format_(core::raw_format::yuyv422),fps_(25)
{
	IOTHREAD_INIT(parameters)

	std::stringstream strs;

	std::string codec = ultragrid::yuri_to_uv_string(format_);
	if (codec.empty()) {
		log[log::fatal] << "Unsupported format requested!";
		throw exception::InitializationFailed("Failed to initialize testcard device!");
	}
	strs << "testcard:" << resolution_.width << ":" << resolution_.height << ":" << fps_ <<":" << codec;
	if (!init_capture(strs.str())) {
		throw exception::InitializationFailed("Failed to initialize testcard device!");
	}
}

UVTestcard::~UVTestcard() noexcept
{
}

bool UVTestcard::set_param(const core::Parameter& param)
{
	if (param.get_name() == "fps") {
		fps_ = param.get<int>();
	} else if (param.get_name() == "format") {
		format_ = core::raw_format::parse_format(param.get<std::string>());
	} else if (param.get_name() == "resolution") {
		resolution_ = param.get<resolution_t>();
	} else return ultragrid::UVVideoSource::set_param(param);
	return true;
}

} /* namespace uv_v4l2 */
} /* namespace yuri */
