/*!
 * @file 		UVRTDxt.cpp
 * @author 		<Your name>
 * @date		17.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "UVRTDxt.h"
#include "yuri/core/Module.h"
#include "yuri/ultragrid/YuriUltragrid.h"
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
	return p;
}


UVRTDxt::UVRTDxt(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
ultragrid::UVVideoCompress(log_,parent, "uv_rtdxt", UV_COMPRESS_DETAIL(dxt_glsl))
{
	IOTHREAD_INIT(parameters)

	if(!init_compressor("DXT1")) {
		log[log::fatal] << "Failed to create encoder";
		throw exception::InitializationFailed("Failed to create DXT encoder");
	}
}

UVRTDxt::~UVRTDxt() noexcept
{
}


bool UVRTDxt::set_param(const core::Parameter& param)
{
	return core::IOThread::set_param(param);
}

} /* namespace uv_rtdxt */
} /* namespace yuri */
