/*!
 * @file 		UVGl.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		17.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "UVGl.h"
#include "yuri/core/Module.h"
extern "C" {
#include "video_display.h"
#include "video_display/gl.h"
}

namespace yuri {
namespace uv_gl {


IOTHREAD_GENERATOR(UVGl)

MODULE_REGISTRATION_BEGIN("uv_gl")
		REGISTER_IOTHREAD("uv_gl",UVGl)
MODULE_REGISTRATION_END()

core::Parameters UVGl::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("UVGl");
	return p;
}


UVGl::UVGl(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
ultragrid::UVVideoSink(log_,parent, "uv_gl", UV_SINK_DETAIL(gl))
{
	IOTHREAD_INIT(parameters)
	if (!init_sink("",0)) {
		log[log::fatal] << "Failed to initialize SDL device";
		throw exception::InitializationFailed("Failed to initialzie SDL device");
	}
}

UVGl::~UVGl() noexcept
{
}
bool UVGl::set_param(const core::Parameter& param)
{
	return core::IOThread::set_param(param);
}

} /* namespace uv_gl */
} /* namespace yuri */
