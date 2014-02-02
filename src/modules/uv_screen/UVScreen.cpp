/*!
 * @file 		UVScreen.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		16.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "UVScreen.h"
#include "yuri/core/Module.h"
extern "C" {
#include "video_capture/screen_x11.h"
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
ultragrid::UVVideoSource(log_,parent,"uv_screen",UV_CAPTURE_DETAIL(screen_x11)),
fps_(30)
{
	IOTHREAD_INIT(parameters)

	std::stringstream strs;
	strs << "screen:fps=" << fps_;

	if (!!init_capture(strs.str())) {
		throw exception::InitializationFailed("Failed to initialize screen capture!");
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
