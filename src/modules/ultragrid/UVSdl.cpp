/*!
 * @file 		UVSdl.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		16.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "UVSdl.h"
#include "yuri/core/Module.h"
extern "C" {
#include "video_display.h"
#include "video_display/sdl.h"
}
namespace yuri {
namespace uv_sdl {


IOTHREAD_GENERATOR(UVSdl)

core::Parameters UVSdl::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("UVSdl");
	p["fullscreen"]["Start in fullscreen"]=false;
	p["deinterlace"]["Enable deinterlacing"]=false;
	return p;
}

UVSdl::UVSdl(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
ultragrid::UVVideoSink(log_, parent, "uv_sdl", UV_SINK_DETAIL(sdl)),
fullscreen_(false),deinterlace_(false)
{
	IOTHREAD_INIT(parameters)
	// UV SDL seems to accept either deinterlace or fullscreen, not both...
	std::string sdl_params = std::string() + (fullscreen_?"fs":(deinterlace_?"d":""));
	log[log::info] << "SDL params: " << sdl_params;
	if (!init_sink(sdl_params,0)) {
		log[log::fatal] << "Failed to initialize SDL device";
		throw exception::InitializationFailed("Failed to initialzie SDL device");
	}
}

UVSdl::~UVSdl() noexcept
{
}

bool UVSdl::set_param(const core::Parameter& param)
{
	if (param.get_name() == "fullscreen") {
		fullscreen_ = param.get<bool>();
	} else if (param.get_name() == "deinterlace") {
		deinterlace_ = param.get<bool>();
	} else return ultragrid::UVVideoSink::set_param(param);
	return true;
}

} /* namespace uv_sdl */
} /* namespace yuri */
