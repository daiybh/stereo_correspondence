/*!
 * @file 		UVDecklink.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		16.06.2014
 * @copyright	CESNET, z.s.p.o, 2014
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "UVDecklink.h"
#include "yuri/core/Module.h"
extern "C" {
#include "video_capture/decklink.h"
}
namespace yuri {
namespace uv_decklink {


IOTHREAD_GENERATOR(UVDecklink)

core::Parameters UVDecklink::configure()
{
	core::Parameters p = ultragrid::UVVideoSource::configure();
	p.set_description("UVDecklink");
	p["device"]["Index of decklink device"]=0;
	p["mode"]["Index of mode to use"]=-1; // This should be done as in yuri decklink, based on mode name..
	p["connection"]["Connection (SDI, HDMI, ...)"]="";
	return p;
}


UVDecklink::UVDecklink(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
ultragrid::UVVideoSource(log_,parent,"uv_decklink",UV_CAPTURE_DETAIL(decklink)),
device_(0),mode_(-1)
{
	IOTHREAD_INIT(parameters)

	std::stringstream strs;
	strs << "decklink:"<< device_;
	if (mode_ >= 0) {
		strs << ":"<<mode_<<":UYVY";
	}
	if (!connection_.empty()) {
		strs << ":connection="<<connection_;
	}

	if (!init_capture(strs.str())) {
		throw exception::InitializationFailed("Failed to initialize decklink capture!");
	}
}

UVDecklink::~UVDecklink() noexcept
{
}

bool UVDecklink::set_param(const core::Parameter& param)
{
	if (param.get_name() == "device") {
		device_ = param.get<int>();
	} else if (param.get_name() == "mode") {
		mode_ = param.get<int>();
	} else if (param.get_name() == "mode") {
		connection_ = param.get<std::string>();
	} else return ultragrid::UVVideoSource::set_param(param);
	return true;
}

} /* namespace uv_v4l2 */
} /* namespace yuri */
