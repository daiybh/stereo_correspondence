/*!
 * @file 		UVDeltaCastDVI.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		12.06.2014
 * @copyright	CESNET, z.s.p.o, 2014
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "UVDeltaCastDVI.h"
#include "yuri/core/Module.h"
extern "C" {
#include "video_capture/deltacast_dvi.h"
}
namespace yuri {
namespace uv_deltacast {


IOTHREAD_GENERATOR(UVDeltaCastDVI)

core::Parameters UVDeltaCastDVI::configure()
{
	core::Parameters p = ultragrid::UVVideoSource::configure();
	p.set_description("UVDeltaCastDVI");
	p["fps"]=30;
	return p;
}


UVDeltaCastDVI::UVDeltaCastDVI(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
ultragrid::UVVideoSource(log_,parent,"uv_deltacast",UV_CAPTURE_DETAIL(deltacast_dvi)),
fps_(30)
{
	IOTHREAD_INIT(parameters)

	std::stringstream strs;
	//strs << /*"screen:"<<*/ "fps=" << fps_;

	if (!init_capture(strs.str())) {
		throw exception::InitializationFailed("Failed to initialize screen capture!");
	}
}

UVDeltaCastDVI::~UVDeltaCastDVI() noexcept
{
}

bool UVDeltaCastDVI::set_param(const core::Parameter& param)
{
	if (param.get_name() == "fps") {
		fps_ = param.get<int>();
	} else return ultragrid::UVVideoSource::set_param(param);
	return true;
}

} /* namespace uv_v4l2 */
} /* namespace yuri */
