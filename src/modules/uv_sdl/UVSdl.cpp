/*!
 * @file 		UVSdl.cpp
 * @author 		<Your name>
 * @date		16.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
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

MODULE_REGISTRATION_BEGIN("uv_sdl")
		REGISTER_IOTHREAD("uv_sdl",UVSdl)
MODULE_REGISTRATION_END()

core::Parameters UVSdl::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("UVSdl");
	return p;
}

UVSdl::UVSdl(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
ultragrid::UVVideoSink(log_, parent, "uv_sdl", UV_SINK_DETAIL(sdl))
{
	IOTHREAD_INIT(parameters)
	if (!init_sink("",0)) {
		log[log::fatal] << "Failed to initialize SDL device";
		throw exception::InitializationFailed("Failed to initialzie SDL device");
	}
}

UVSdl::~UVSdl() noexcept
{
}

bool UVSdl::set_param(const core::Parameter& param)
{
	return ultragrid::UVVideoSink::set_param(param);
}

} /* namespace uv_sdl */
} /* namespace yuri */
