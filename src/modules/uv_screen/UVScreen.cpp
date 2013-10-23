/*!
 * @file 		UVScreen.cpp
 * @author 		<Your name>
 * @date		16.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "UVScreen.h"
#include "yuri/core/Module.h"
extern "C" {
#include "video_capture/screen.h"
}
namespace yuri {
namespace uv_v4l2 {


IOTHREAD_GENERATOR(UVScreen)

MODULE_REGISTRATION_BEGIN("uv_screen")
		REGISTER_IOTHREAD("uv_screen",UVScreen)
MODULE_REGISTRATION_END()

core::Parameters UVScreen::configure()
{
	core::Parameters p = ultragrid::UVVideoSource::configure();
	p.set_description("UVScreen");
	p["fps"]=30;
	return p;
}


UVScreen::UVScreen(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
ultragrid::UVVideoSource(log_,parent,"uv_screen",UV_CAPTURE_DETAIL(screen)),
fps_(30)
{
	IOTHREAD_INIT(parameters)

	std::stringstream strs;
	strs << "screen:fps=" << fps_;

	if (!!init_capture(strs.str())) {
		throw exception::InitializationFailed("Failed to initialize v4l2 device!");
	}
}

UVScreen::~UVScreen() noexcept
{
}

bool UVScreen::set_param(const core::Parameter& param)
{
	if (param.get_name() == "fps") {
		fps_ = param.get<int>();
	} else return ultragrid::UVVideoSource::set_param(param);
	return true;
}

} /* namespace uv_v4l2 */
} /* namespace yuri */
